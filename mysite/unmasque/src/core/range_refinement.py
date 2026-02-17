import copy
import time
from typing import List, Tuple, Optional

from ..core.abstract.MutationPipeLineBase import MutationPipeLineBase
from ..core.abstract.abstractConnection import AbstractConnectionHelper
from ..core.db_restorer import DbRestorer
from ..util.aoa_utils import get_constants_for
from ..util.utils import get_format, get_val_plus_delta, get_mid_val, get_min_and_max_val, get_cast_value
from ..util.constants import NON_TEXT_TYPES


class RangeRefinement(MutationPipeLineBase):
    """
    Detects and removes 'holes' inside over-approximated disjunction ranges.

    Problem: Binary search on d_min merges non-contiguous ranges like
    [10,20] U [30,40] into [10,40] because it cannot detect the gap [21,29].

    Algorithm (R_E - R_H approach):
        1. Restore full DB. For each range predicate, sample actual DB values
           within the range and test them on d_min against Q_H.
        2. If all sampled values are accepted -> ranges are exact, done.
        3. A value that makes Q_H empty when placed in d_min is a gap witness.
        4. Per-attribute mutation on d_min confirms attribution.
        5. Binary search for the exact hole boundaries within the attributed range,
           then split the range predicate into sub-ranges excluding the hole.
        6. Repeat until no more gap witnesses are found.
    """

    REFINEMENT_CUTOFF = 20  # max iterations to prevent infinite loops

    def __init__(self, connectionHelper: AbstractConnectionHelper,
                 core_relations: List[str],
                 global_min_instance_dict: dict,
                 filter_extractor,
                 all_sizes: dict):
        super().__init__(connectionHelper, core_relations, global_min_instance_dict,
                         "RangeRefinement")
        self.filter_extractor = filter_extractor
        self.all_sizes = all_sizes
        self.db_restorer = DbRestorer(self.connectionHelper, self.core_relations)
        self.db_restorer.set_all_sizes(self.all_sizes)

    def extract_params_from_args(self, args):
        return args[0], args[1], args[2]

    def doActualJob(self, args=None):
        """
        Entry point.
        args: (query, arithmetic_filters, or_predicates)
        Returns: (refined_arithmetic_filters, refined_or_predicates) or None on error.
        """
        query, arithmetic_filters, or_predicates = self.extract_params_from_args(args)
        refined_filters, refined_or = self.refine(query, arithmetic_filters, or_predicates)
        return refined_filters, refined_or

    def refine(self, query, arithmetic_filters, or_predicates):
        """
        Main refinement loop. Iteratively discovers holes in over-approximated
        range predicates and splits them.
        """
        current_filters = copy.deepcopy(arithmetic_filters)
        current_or = copy.deepcopy(or_predicates)

        for iteration in range(self.REFINEMENT_CUTOFF):
            self.logger.debug(f"Range refinement iteration {iteration + 1}")

            # Step 1: Find gap witnesses by sampling DB values within each range
            gap_witness = self._find_first_gap_witness(query, current_filters, current_or)

            if gap_witness is None:
                self.logger.info(f"No gap witnesses found. Ranges are exact after "
                                 f"{iteration} refinement iterations.")
                return current_filters, current_or

            tab, attrib, hole_val = gap_witness
            self.logger.debug(f"Gap witness found: {tab}.{attrib} = {hole_val}")

            # Step 2: Binary search for hole boundaries and split the range
            current_filters, current_or = self._find_hole_and_split(
                query, tab, attrib, hole_val, current_filters, current_or)

        self.logger.info(f"Reached refinement cutoff ({self.REFINEMENT_CUTOFF}).")
        return current_filters, current_or

    def _find_first_gap_witness(self, query, arithmetic_filters, or_predicates):
        """
        For each range predicate, sample actual values from the full DB within
        the range and test on d_min. Return the first (tab, attrib, val) where
        Q_H returns empty.
        """
        # Collect all range predicates: from arithmetic_filters + or_predicates
        range_preds = self._collect_all_range_preds(arithmetic_filters, or_predicates)
        if not range_preds:
            return None

        for tab, attrib, lb, ub in range_preds:
            datatype = self.filter_extractor.get_datatype((tab, attrib))
            witness = self._sample_for_gap(tab, attrib, datatype, lb, ub, query)
            if witness is not None:
                return (tab, attrib, witness)

        return None

    def _collect_all_range_preds(self, arithmetic_filters, or_predicates):
        """Collect (tab, attrib, lb, ub) for all range predicates."""
        preds = []
        for p in arithmetic_filters:
            if len(p) >= 5 and p[2] == 'range':
                preds.append((p[0], p[1], p[3], p[4]))
        for group in or_predicates:
            for pred in group:
                if pred and len(pred) >= 5 and pred[2] == 'range':
                    key = (pred[0], pred[1], pred[3], pred[4])
                    if key not in preds:
                        preds.append(key)
        return preds

    def _sample_for_gap(self, tab, attrib, datatype, lb, ub, query):
        """
        Query distinct values of tab.attrib from full DB within [lb, ub].
        For each sampled value, set d_min to that value and run Q_H.
        Return the first value where Q_H is empty (gap witness), or None.
        """
        # Restore full DB for sampling query
        if not self._restore_full_db():
            return None

        val_lb = get_format(datatype, lb)
        val_ub = get_format(datatype, ub)
        sample_query = (
            f"SELECT DISTINCT {attrib} FROM {self.get_original_table_name(tab)} "
            f"WHERE {attrib} >= {val_lb} AND {attrib} <= {val_ub} "
            f"ORDER BY {attrib}"
        )
        try:
            res, desc = self.connectionHelper.execute_sql_fetchall(sample_query, self.logger)
        except Exception as e:
            self.logger.error(f"Sampling query failed: {e}")
            return None

        if res is None or len(res) == 0:
            return None

        all_values = [row[0] for row in res]
        self.logger.debug(f"Sampled {len(all_values)} distinct values for "
                          f"{tab}.{attrib} in [{lb}, {ub}]")

        # Now test on d_min: restore d_min, mutate this attrib, run Q_H
        for val in all_values:
            self._restore_d_min()
            self._update_attrib_val(tab, attrib, datatype, val)
            result = self.app.doJob(query)
            if self.app.isQ_result_empty(result):
                self._restore_d_min()
                return val

        self._restore_d_min()
        return None

    def _restore_full_db(self):
        """Restore all core_relations to full database."""
        try:
            for tab in self.core_relations:
                row_count = self.db_restorer.restore_table_and_confirm(tab)
                if not row_count:
                    self.logger.error(f"Could not restore {tab}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"DB restore failed: {e}")
            return False

    def _restore_d_min(self):
        """Restore d_min tables from the stored dict."""
        for tab in self.core_relations:
            values = self.global_min_instance_dict[tab]
            attribs, vals = values[0], values[1]
            self.connectionHelper.execute_sql([
                self.connectionHelper.queries.truncate_table(
                    self.get_fully_qualified_table_name(tab))])
            attrib_list = ", ".join(attribs)
            self.connectionHelper.execute_sql_with_params(
                self.connectionHelper.queries.insert_into_tab_attribs_format(
                    f"({attrib_list})", "",
                    self.get_fully_qualified_table_name(tab)),
                [vals])

    def _update_attrib_val(self, tab, attrib, datatype, val):
        """Update a single attribute value in d_min."""
        if datatype in NON_TEXT_TYPES:
            update_q = self.connectionHelper.queries.update_tab_attrib_with_value(
                self.get_fully_qualified_table_name(tab), attrib,
                get_format(datatype, val))
        else:
            update_q = self.connectionHelper.queries.update_tab_attrib_with_quoted_value(
                self.get_fully_qualified_table_name(tab), attrib, val)
        self.connectionHelper.execute_sql([update_q])

    def _find_hole_and_split(self, query, tab, attrib, hole_val,
                              arithmetic_filters, or_predicates):
        """
        Binary search for exact hole boundaries within the range
        containing hole_val, then split the range predicate.
        """
        datatype = self.filter_extractor.get_datatype((tab, attrib))
        delta, _ = get_constants_for(datatype)

        # Find the range predicate that contains hole_val
        pred_idx, pred, source = self._find_containing_range(
            tab, attrib, hole_val, arithmetic_filters, or_predicates)

        if pred is None:
            self.logger.error(f"Cannot find range containing {hole_val} for {tab}.{attrib}")
            return copy.deepcopy(arithmetic_filters), copy.deepcopy(or_predicates)

        lb, ub = pred[3], pred[4]

        # Binary search for upper edge of left sub-range
        # (transition from accepted to rejected, going from lb toward hole_val)
        hole_start = self._binary_search_boundary(
            query, tab, attrib, datatype, lb, hole_val, direction='find_rejection_start')

        # Binary search for lower edge of right sub-range
        # (transition from rejected to accepted, going from hole_val toward ub)
        hole_end = self._binary_search_boundary(
            query, tab, attrib, datatype, hole_val, ub, direction='find_acceptance_start')

        self.logger.debug(f"Hole boundaries: {tab}.{attrib} gap = [{hole_start}, {hole_end}]")

        # Compute sub-ranges excluding the hole
        left_ub = get_val_plus_delta(datatype, hole_start, -1 * delta)
        right_lb = get_val_plus_delta(datatype, hole_end, 1 * delta)

        sub_ranges = []
        if lb <= left_ub:
            sub_ranges.append((tab, attrib, 'range', lb, left_ub))
        if right_lb <= ub:
            sub_ranges.append((tab, attrib, 'range', right_lb, ub))

        if source == 'arithmetic':
            return self._apply_split_to_arithmetic(
                pred_idx, pred, sub_ranges, arithmetic_filters, or_predicates)
        else:
            return self._apply_split_to_or(
                pred_idx, pred, sub_ranges, arithmetic_filters, or_predicates, source)

    def _find_containing_range(self, tab, attrib, val,
                                arithmetic_filters, or_predicates):
        """
        Find the range predicate containing val.
        Returns (index, predicate_tuple, source_type).
        source_type is 'arithmetic' or ('or', group_idx, pred_idx).
        """
        for i, pred in enumerate(arithmetic_filters):
            if (pred[0] == tab and pred[1] == attrib
                    and pred[2] == 'range' and pred[3] <= val <= pred[4]):
                return i, pred, 'arithmetic'

        for g_idx, group in enumerate(or_predicates):
            for p_idx, pred in enumerate(group):
                if (pred and len(pred) >= 5 and pred[0] == tab
                        and pred[1] == attrib and pred[2] == 'range'
                        and pred[3] <= val <= pred[4]):
                    return p_idx, pred, ('or', g_idx, p_idx)

        return -1, None, None

    def _binary_search_boundary(self, query, tab, attrib, datatype,
                                 search_lb, search_ub, direction):
        """
        Binary search for hole boundary.

        direction='find_rejection_start':
            Find the smallest value in [search_lb, search_ub] where Q_H is empty.
            (Left boundary of the hole)

        direction='find_acceptance_start':
            Find the largest value in [search_lb, search_ub] where Q_H is empty.
            (Right boundary of the hole)
        """
        delta, cutoff = get_constants_for(datatype)
        low = get_cast_value(datatype, search_lb)
        high = get_cast_value(datatype, search_ub)

        while self._is_range_wide_enough(datatype, low, high, cutoff):
            mid = get_mid_val(datatype, high, low)
            if mid == low or mid == high:
                break

            self._restore_d_min()
            self._update_attrib_val(tab, attrib, datatype, mid)
            result = self.app.doJob(query)
            is_empty = self.app.isQ_result_empty(result)

            if direction == 'find_rejection_start':
                # Looking for leftmost rejected value
                if is_empty:
                    high = mid  # hole includes mid, search left
                else:
                    low = mid  # still accepted, hole starts right of mid
            else:  # find_acceptance_start
                # Looking for rightmost rejected value
                if is_empty:
                    low = mid  # still in hole, search right
                else:
                    high = mid  # accepted, hole ends left of mid

        self._restore_d_min()

        if direction == 'find_rejection_start':
            return high  # first rejected value
        else:
            return low  # last rejected value

    def _is_range_wide_enough(self, datatype, low, high, cutoff):
        """Check if binary search should continue."""
        if datatype == 'date':
            return int((high - low).days) >= cutoff
        else:
            return (high - low) >= cutoff

    def _apply_split_to_arithmetic(self, pred_idx, old_pred, sub_ranges,
                                    arithmetic_filters, or_predicates):
        """Apply the split to an arithmetic_filters range predicate.

        Key invariant: all sub-ranges must end up in a SINGLE multi-element
        or_predicates group so GenPipelineContext emits (... OR ...).
        Old single-element groups referencing the over-approximated range
        are purged to prevent __generate_arithmetic_conjunctive_disjunctions
        from re-introducing the stale range as a conjunctive predicate.
        """
        new_filters = copy.deepcopy(arithmetic_filters)
        new_or = copy.deepcopy(or_predicates)

        # Remove the original over-approximated range from arithmetic_filters
        new_filters.remove(old_pred)

        if len(sub_ranges) == 0:
            # Hole covers entire range — shouldn't happen, re-add original
            self.logger.warning("Hole covers entire range, keeping original predicate.")
            new_filters.append(old_pred)
        elif len(sub_ranges) == 1:
            # Only one sub-range remains — just a narrower range
            new_filters.append(sub_ranges[0])
        else:
            # Two sub-ranges: this creates a disjunction (OR)
            # Add the first sub-range to arithmetic_filters
            new_filters.append(sub_ranges[0])
            # Create an or_predicates entry with both sub-ranges
            or_entry = tuple(sub_ranges)
            new_or.append(or_entry)

        # Purge old single-element or_predicates groups that still
        # reference the over-approximated range.  Without this cleanup,
        # GenPipelineContext.__generate_arithmetic_conjunctive_disjunctions
        # re-adds the stale range as a redundant AND predicate.
        tab, attrib = old_pred[0], old_pred[1]
        old_lb, old_ub = old_pred[3], old_pred[4]
        purged_or = []
        for group in new_or:
            non_empty = [p for p in group if p]
            has_old = any(
                p and len(p) >= 5
                and p[0] == tab and p[1] == attrib
                and p[2] == 'range'
                and p[3] == old_lb and p[4] == old_ub
                for p in group
            )
            if has_old and len(non_empty) <= 1:
                # Single-element group with the stale range -> drop it
                continue
            if has_old and len(non_empty) > 1:
                # Multi-element group -> replace old pred with first sub-range
                new_tuple = []
                for p in group:
                    if (p and len(p) >= 5
                            and p[0] == tab and p[1] == attrib
                            and p[2] == 'range'
                            and p[3] == old_lb and p[4] == old_ub):
                        new_tuple.append(sub_ranges[0] if sub_ranges else p)
                    else:
                        new_tuple.append(p)
                purged_or.append(tuple(new_tuple))
            else:
                purged_or.append(group)
        new_or = purged_or

        self.logger.debug(f"Split arithmetic predicate: {old_pred} -> {sub_ranges}")
        return new_filters, new_or

    def _apply_split_to_or(self, pred_idx, old_pred, sub_ranges,
                            arithmetic_filters, or_predicates, source_info):
        """Apply the split to an or_predicates range predicate."""
        new_filters = copy.deepcopy(arithmetic_filters)
        new_or = copy.deepcopy(or_predicates)
        g_idx, p_idx = source_info[1], source_info[2]

        # Get the group and replace the old pred with sub-ranges
        group = list(new_or[g_idx])
        group.pop(p_idx)
        for i, sr in enumerate(sub_ranges):
            group.insert(p_idx + i, sr)
        new_or[g_idx] = tuple(group)

        self.logger.debug(f"Split OR predicate at group {g_idx}: {old_pred} -> {sub_ranges}")
        return new_filters, new_or
