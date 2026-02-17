import time
from abc import ABC, abstractmethod

from ....src.core.range_refinement import RangeRefinement
from ....src.pipeline.abstract.generic_pipeline import GenericPipeLine
from ....src.util.constants import RANGE_REFINEMENT, DONE, RUNNING, START, ERROR


class RangeRefinementPipeLine(GenericPipeLine, ABC):
    """
    Pipeline mixin that refines over-approximated disjunction ranges.

    Placed between DisjunctionPipeLine._extract_disjunction() and
    the GenPipelineContext preprocessing step. Detects holes in extracted
    ranges by sampling real DB values and testing them on d_min, then
    splits ranges using binary search for exact hole boundaries.
    """

    def __init__(self, connectionHelper, name="Base Pipeline"):
        super().__init__(connectionHelper, name)

    @abstractmethod
    def extract(self, query):
        pass

    @abstractmethod
    def process(self, query: str):
        raise NotImplementedError("Trouble!")

    @abstractmethod
    def doJob(self, query, qe=None):
        raise NotImplementedError("Trouble!")

    @abstractmethod
    def verify_correctness(self, query, result):
        raise NotImplementedError("Trouble!")

    def _refine_disjunction_ranges(self, query, core_relations, time_profile):
        """
        Refine over-approximated range predicates by detecting and removing holes.

        Requires self.aoa.arithmetic_filters and self.or_predicates to be populated
        (i.e., _mutation_pipeline and _extract_disjunction must have run first).

        Returns: (success: bool, time_profile)
        """
        if not self.connectionHelper.config.detect_or:
            self.logger.info("OR detection disabled — skipping range refinement.")
            return True, time_profile

        # Check if there are any range predicates to refine
        has_ranges = any(p[2] == 'range' for p in self.aoa.arithmetic_filters if len(p) >= 5)
        if not has_ranges:
            # Also check or_predicates
            for group in self.or_predicates:
                for pred in group:
                    if pred and len(pred) >= 5 and pred[2] == 'range':
                        has_ranges = True
                        break
                if has_ranges:
                    break

        if not has_ranges:
            self.logger.info("No range predicates found — skipping range refinement.")
            return True, time_profile

        self.logger.info("Starting range refinement phase...")
        self.update_state(RANGE_REFINEMENT + START)

        try:
            refiner = RangeRefinement(
                self.connectionHelper,
                core_relations,
                self.global_min_instance_dict,
                self.filter_extractor,
                self.all_sizes)

            self.update_state(RANGE_REFINEMENT + RUNNING)
            start_t = time.time()

            result = refiner.doJob(query, self.aoa.arithmetic_filters, self.or_predicates)

            end_t = time.time()
            elapsed = end_t - start_t

            if result is None:
                self.logger.error("Range refinement returned None.")
                self.update_state(ERROR)
                time_profile.update_for_where_clause(elapsed, refiner.app_calls)
                return False, time_profile

            refined_filters, refined_or = result

            # Update the pipeline state with refined predicates
            self.aoa.arithmetic_filters = refined_filters
            self.or_predicates = refined_or

            self.update_state(RANGE_REFINEMENT + DONE)
            time_profile.update_for_where_clause(elapsed, refiner.app_calls)

            self.logger.info(f"Range refinement completed in {elapsed:.2f}s. "
                             f"Filters: {len(refined_filters)}, "
                             f"OR groups: {len(refined_or)}")
            return True, time_profile

        except Exception as e:
            self.logger.error(f"Range refinement failed with error: {e}")
            self.update_state(ERROR)
            return False, time_profile
