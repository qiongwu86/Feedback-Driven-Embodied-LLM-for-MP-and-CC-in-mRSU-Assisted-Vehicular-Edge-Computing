from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import pickle

import numpy as np

from mobility.vehicle_mobility import OneDimensionalVehicleMobility, VehicleState
from simulation.config import MRSUSimulationConfig
from simulation.coverage import CoverageModel, CoverageSnapshot
from simulation.metrics import RoundMetrics, evaluate_cache_hit_ratio


@dataclass
class RSUState:
    position: float
    speed: float = 0.0


class MRSUEnvironment:
    """One-dimensional VEC environment with ordinary vehicles, one mRSU, one fRSU."""

    columns = ["user_id", "movie_id", "rating", "gender", "age", "occupation"]

    def __init__(self, config: MRSUSimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.train_data, self.test_data = load_preprocessed_movielens(
            Path(__file__).resolve().parents[1] / "datasets", "ml1m"
        )
        self.train_rows = [
            _normalize_row(row)
            for row in self.train_data
            if int(row[1]) <= config.movie_num
        ]
        self.test_rows = [
            _normalize_row(row)
            for row in self.test_data
            if int(row[1]) <= config.movie_num
        ]

        self.user_ids = _first_unique_users(self.test_rows, config.user_num)
        if not self.user_ids:
            raise RuntimeError("No MovieLens users found for mRSU simulation.")
        user_set = set(self.user_ids)
        self.test_rows = [row for row in self.test_rows if row[0] in user_set]
        self.train_rows = [row for row in self.train_rows if row[0] in user_set]

        self.mobility = OneDimensionalVehicleMobility(
            road_length=config.road_length,
            vehicle_num=config.vehicle_num,
            user_ids=self.user_ids,
            min_speed=config.min_vehicle_speed,
            max_speed=config.max_vehicle_speed,
            rng=self.rng,
            speed_noise_std=config.vehicle_speed_noise_std,
            platoon_cluster_count=config.platoon_cluster_count,
            platoon_cluster_std=config.platoon_cluster_std,
            platoon_speed_std=config.platoon_speed_std,
        )
        self.user_to_vehicle = self.mobility.user_to_vehicle()
        self.mrsu = RSUState(
            position=config.mrsu_initial_position,
            speed=config.mrsu_initial_speed,
        )
        self.frsu = RSUState(position=config.frsu_position, speed=0.0)
        self.mrsu_cache: List[int] = []
        self.frsu_cache: List[int] = []
        self.last_metrics = RoundMetrics(0, 0, 0.0, 0, 0, 0, 0, 0)
        self.last_round_missed_counter = Counter()

        self.global_popularity = self._build_popularity(self.train_rows)
        self.global_top_contents = [content for content, _ in self.global_popularity.most_common()]
        self.user_profiles = self._build_user_profiles()
        self.user_test_profiles = self._build_user_test_profiles()
        self.vehicle_prediction_profiles = self._build_vehicle_prediction_profiles()
        self.vehicle_true_profiles = self._build_vehicle_true_profiles()
        self.vehicle_prediction_states = _copy_counter_dict(self.vehicle_prediction_profiles)
        self.vehicle_true_states = _copy_counter_dict(self.vehicle_true_profiles)
        self.predicted_request_count_state = self._initial_request_count_state(
            self.vehicle_prediction_profiles
        )
        self.true_request_count_state = self._initial_request_count_state(
            self.vehicle_true_profiles
        )
        self.user_history = self._build_user_history()
        self.last_round_vehicle_missed_counters: Dict[int, Counter] = defaultdict(Counter)

    def _build_popularity(self, rows: List[Tuple[int, int, int, float, int, int]]) -> Counter:
        counter = Counter()
        for row in rows:
            counter[int(row[1])] += max(1, int(row[2]))
        return counter

    def _build_user_profiles(self) -> Dict[int, Counter]:
        profiles: Dict[int, Counter] = defaultdict(Counter)
        for user_id, movie_id, rating, *_ in self.train_rows:
            profiles[int(user_id)][int(movie_id)] += max(1, int(rating))
        return {int(user_id): Counter(profile) for user_id, profile in profiles.items()}

    def _build_user_test_profiles(self) -> Dict[int, Counter]:
        profiles: Dict[int, Counter] = defaultdict(Counter)
        for user_id, movie_id, rating, *_ in self.test_rows:
            profiles[int(user_id)][int(movie_id)] += max(1, int(rating))
        return {int(user_id): Counter(profile) for user_id, profile in profiles.items()}

    def _build_vehicle_prediction_profiles(self) -> Dict[int, Counter]:
        profiles: Dict[int, Counter] = {}
        global_fallback = Counter(dict(self.global_popularity.most_common(300)))
        for vehicle in self.mobility.vehicles:
            profile = Counter(self.user_profiles.get(int(vehicle.user_id), Counter()))
            if not profile:
                profile.update(global_fallback)
            profiles[int(vehicle.vehicle_id)] = profile
        return profiles

    def _build_vehicle_true_profiles(self) -> Dict[int, Counter]:
        profiles: Dict[int, Counter] = {}
        global_test_fallback = self._global_test_popularity()
        if not global_test_fallback:
            global_test_fallback.update(Counter(dict(self.global_popularity.most_common(300))))
        for vehicle in self.mobility.vehicles:
            profile = Counter(self.user_test_profiles.get(int(vehicle.user_id), Counter()))
            if not profile:
                profile.update(global_test_fallback)
            profiles[int(vehicle.vehicle_id)] = profile
        return profiles

    def _global_test_popularity(self) -> Counter:
        counter = Counter()
        for _, movie_id, rating, *_ in self.test_rows:
            counter[int(movie_id)] += max(1, int(rating))
        return counter

    def _initial_request_count_state(self, vehicle_profiles: Dict[int, Counter]) -> Dict[int, float]:
        vehicles = list(self.mobility.vehicles)
        target = float(round((self.config.request_min + self.config.request_max) / 2.0))
        weights = np.array(
            [
                max(1.0, float(sum(vehicle_profiles.get(v.vehicle_id, Counter()).values())))
                for v in vehicles
            ],
            dtype=float,
        )
        weights = weights / max(float(weights.sum()), 1e-9)
        return {
            int(vehicle.vehicle_id): float(target * weight)
            for vehicle, weight in zip(vehicles, weights)
        }

    def _build_user_history(self, max_items_per_user: int = 40) -> Dict[int, List[Tuple[int, int]]]:
        grouped: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for user_id, movie_id, rating, *_ in self.train_rows:
            grouped[int(user_id)].append((int(movie_id), int(rating)))
        history = {}
        for uid, items in grouped.items():
            items.sort(key=lambda item: (-item[1], item[0]))
            history[int(uid)] = items[:max_items_per_user]
        return history

    def vehicle_states(self) -> List[VehicleState]:
        return list(self.mobility.vehicles)

    def step_mobility(self) -> None:
        self.mobility.step(self.config.dt)

    def decision_window_ticks(self, decision_index: int) -> int:
        start_tick = int(decision_index) * int(self.config.decision_interval)
        remaining = int(self.config.rounds) - start_tick
        return max(1, min(int(self.config.decision_interval), max(1, remaining)))

    def sample_round_requests(self, round_index: int) -> Dict[int, List[int]]:
        """Sample one round of evaluation requests from per-vehicle test profiles.

        The concrete movies come from each bound user's held-out MovieLens test
        profile. Smoothed count and demand states keep demand evolution
        temporally local instead of redrawing the whole road demand every round.
        """

        target = int(self.rng.integers(self.config.request_min, self.config.request_max + 1))
        allocations = self._smooth_request_allocations(
            state=self.true_request_count_state,
            vehicle_profiles=self.vehicle_true_profiles,
            target=target,
            rng=self.rng,
            noise_scale=self.config.true_demand_noise_scale,
        )
        self._smooth_vehicle_demand_states(
            states=self.vehicle_true_states,
            base_profiles=self.vehicle_true_profiles,
            feedback_profiles={},
            feedback_weight=0.0,
            prior_weight=self.config.true_demand_prior_weight,
            rng=self.rng,
            noise_scale=self.config.true_demand_noise_scale,
        )

        vehicle_requests: Dict[int, List[int]] = {}
        for vehicle in self.mobility.vehicles:
            vehicle_id = int(vehicle.vehicle_id)
            vehicle_requests[vehicle_id] = self._sample_from_profile(
                self.vehicle_true_states.get(vehicle_id, Counter()),
                int(allocations.get(vehicle_id, 0)),
                self.rng,
            )
        return dict(vehicle_requests)

    def predict_round_requests(self, round_index: int, use_feedback: bool = True) -> Dict[int, List[int]]:
        """Build a smoothed demand signal without looking at current test requests.

        Prediction states are initialized from MovieLens training profiles and
        can optionally be updated with previous-round vehicle-level miss feedback.
        A separate deterministic RNG keeps prediction generation from changing the
        true evaluation request stream.
        """

        target = int(round((self.config.request_min + self.config.request_max) / 2.0))
        rng = np.random.default_rng(int(self.config.seed) + 104729 + int(round_index))
        allocations = self._smooth_request_allocations(
            state=self.predicted_request_count_state,
            vehicle_profiles=self.vehicle_prediction_profiles,
            target=target,
            rng=rng,
            noise_scale=self.config.prediction_noise_scale,
        )
        if use_feedback:
            feedback_profiles = {
                int(vehicle_id): Counter(counter)
                for vehicle_id, counter in self.last_round_vehicle_missed_counters.items()
            }
            feedback_weight = self.config.prediction_feedback_weight
        else:
            feedback_profiles = {}
            feedback_weight = 0.0
        self._smooth_vehicle_demand_states(
            states=self.vehicle_prediction_states,
            base_profiles=self.vehicle_prediction_profiles,
            feedback_profiles=feedback_profiles,
            feedback_weight=feedback_weight,
            prior_weight=self.config.prediction_prior_weight,
            rng=rng,
            noise_scale=self.config.prediction_noise_scale,
        )

        predicted_requests: Dict[int, List[int]] = {}
        for vehicle in self.mobility.vehicles:
            vehicle_id = int(vehicle.vehicle_id)
            predicted_requests[vehicle_id] = self._expected_requests_from_profile(
                self.vehicle_prediction_states.get(vehicle_id, Counter()),
                int(allocations.get(vehicle_id, 0)),
            )
        return predicted_requests

    def _smooth_request_allocations(
        self,
        state: Dict[int, float],
        vehicle_profiles: Dict[int, Counter],
        target: int,
        rng: np.random.Generator,
        noise_scale: float,
    ) -> Dict[int, int]:
        vehicles = list(self.mobility.vehicles)
        base_weights = np.array(
            [
                max(1.0, float(sum(vehicle_profiles.get(v.vehicle_id, Counter()).values())))
                for v in vehicles
            ],
            dtype=float,
        )
        if noise_scale > 0:
            noise = rng.normal(loc=1.0, scale=float(noise_scale), size=len(base_weights))
            base_weights = base_weights * np.clip(noise, 0.1, None)
        base_weights = base_weights / max(float(base_weights.sum()), 1e-9)
        desired = np.array([target * weight for weight in base_weights], dtype=float)
        previous = np.array(
            [
                float(state.get(int(vehicle.vehicle_id), desired[idx]))
                for idx, vehicle in enumerate(vehicles)
            ],
            dtype=float,
        )
        alpha = float(np.clip(self.config.request_count_smoothing, 0.0, 0.999))
        smoothed = alpha * previous + (1.0 - alpha) * desired
        smoothed = target * smoothed / max(float(smoothed.sum()), 1e-9)
        for vehicle, value in zip(vehicles, smoothed):
            state[int(vehicle.vehicle_id)] = float(value)
        return _integer_allocations(
            {int(vehicle.vehicle_id): float(value) for vehicle, value in zip(vehicles, smoothed)},
            int(target),
        )

    def _smooth_vehicle_demand_states(
        self,
        states: Dict[int, Counter],
        base_profiles: Dict[int, Counter],
        feedback_profiles: Dict[int, Counter],
        feedback_weight: float,
        prior_weight: float,
        rng: np.random.Generator,
        noise_scale: float,
    ) -> None:
        alpha = float(np.clip(self.config.demand_profile_smoothing, 0.0, 0.999))
        prior_weight = max(0.0, float(prior_weight))
        feedback_weight = max(0.0, float(feedback_weight))
        for vehicle in self.mobility.vehicles:
            vehicle_id = int(vehicle.vehicle_id)
            base = Counter(base_profiles.get(vehicle_id, Counter()))
            feedback = Counter(feedback_profiles.get(vehicle_id, Counter()))
            if not base and not feedback:
                base.update(Counter(dict(self.global_popularity.most_common(300))))

            previous = Counter(states.get(vehicle_id, Counter()))
            keys = set(previous.keys()).union(base.keys()).union(feedback.keys())
            updated = Counter()
            for content_id in keys:
                value = (
                    alpha * float(previous.get(content_id, 0.0))
                    + prior_weight * float(base.get(content_id, 0.0))
                    + feedback_weight * max(0.0, float(feedback.get(content_id, 0.0)))
                )
                if noise_scale > 0:
                    value *= max(0.1, float(rng.normal(loc=1.0, scale=float(noise_scale))))
                if value > 1e-9:
                    updated[int(content_id)] = value
            states[vehicle_id] = updated

    def _sample_from_profile(self, profile: Counter, count: int, rng: np.random.Generator) -> List[int]:
        if count <= 0:
            return []
        items = [
            (int(content_id), float(weight))
            for content_id, weight in profile.items()
            if int(content_id) <= self.config.movie_num and float(weight) > 0
        ]
        if not items:
            items = [(int(content_id), 1.0) for content_id in self.global_top_contents[:300]]
        if not items:
            return []
        content_ids = [content_id for content_id, _ in items]
        weights = np.array([weight for _, weight in items], dtype=float)
        weights = weights / weights.sum()
        sampled = rng.choice(len(content_ids), size=int(count), replace=int(count) > len(content_ids), p=weights)
        return [int(content_ids[int(idx)]) for idx in sampled]

    def _expected_requests_from_profile(self, profile: Counter, count: int) -> List[int]:
        if count <= 0:
            return []
        items = [
            (int(content_id), float(weight))
            for content_id, weight in profile.items()
            if int(content_id) <= self.config.movie_num and float(weight) > 0
        ]
        if not items:
            items = [(int(content_id), 1.0) for content_id in self.global_top_contents[:300]]
        if not items:
            return []

        total_weight = max(sum(weight for _, weight in items), 1e-9)
        expected = {
            int(content_id): float(count) * float(weight) / total_weight
            for content_id, weight in items
        }
        allocations = _integer_allocations(expected, int(count))
        weight_by_content = {int(content_id): float(weight) for content_id, weight in items}
        requests: List[int] = []
        for content_id, allocated in sorted(
            allocations.items(),
            key=lambda item: (int(item[1]), weight_by_content.get(int(item[0]), 0.0), -int(item[0])),
            reverse=True,
        ):
            requests.extend([int(content_id)] * int(allocated))
        return requests[: int(count)]

    def vehicle_request_counters(self, vehicle_requests: Dict[int, List[int]]) -> Dict[int, Counter]:
        return {
            vehicle_id: Counter(int(content_id) for content_id in requests)
            for vehicle_id, requests in vehicle_requests.items()
        }

    def coverage_snapshot(self, mrsu_position: float = None) -> CoverageSnapshot:
        position = self.mrsu.position if mrsu_position is None else mrsu_position
        return CoverageModel.snapshot(
            vehicle_positions=self.mobility.positions(),
            mrsu_position=float(position) % float(self.config.road_length),
            mrsu_radius=self.config.mrsu_radius,
            frsu_position=self.frsu.position,
            frsu_radius=self._effective_frsu_radius(),
            road_length=self.config.road_length,
        )

    def project_service_window_coverage(
        self,
        path_plan=None,
        ticks: int = None,
        include_current: bool = True,
        hold_mrsu_position: bool = False,
    ) -> CoverageSnapshot:
        """Estimate the vehicles covered at least once during a decision window.

        This projection uses current vehicle speeds without stochastic speed noise
        so it can be used as decision-time evidence without advancing the simulator.
        """

        ticks = int(self.config.decision_interval if ticks is None else ticks)
        vehicle_positions = self.mobility.positions().astype(float)
        vehicle_speeds = self.mobility.speeds().astype(float)
        mrsu_position = float(self.mrsu.position)
        mrsu_speed = float(self.mrsu.speed)
        mrsu_seen = set()
        frsu_seen = set()

        if include_current:
            self._accumulate_coverage_sets(vehicle_positions, mrsu_position, mrsu_seen, frsu_seen)

        for tick in range(max(0, ticks)):
            if path_plan is not None and tick + 1 < len(getattr(path_plan, "positions", [])):
                mrsu_position = float(path_plan.positions[tick + 1])
                if tick < len(getattr(path_plan, "velocities", [])):
                    mrsu_speed = float(path_plan.velocities[tick])
            elif not hold_mrsu_position:
                mrsu_position = (mrsu_position + mrsu_speed * float(self.config.dt)) % float(
                    self.config.road_length
                )
            vehicle_positions = (
                vehicle_positions + vehicle_speeds * float(self.config.dt)
            ) % float(self.config.road_length)
            self._accumulate_coverage_sets(vehicle_positions, mrsu_position, mrsu_seen, frsu_seen)

        return self._coverage_from_sets(mrsu_seen, frsu_seen)

    def execute_service_window(
        self,
        path_plan=None,
        ticks: int = None,
        include_current: bool = True,
        hold_mrsu_position: bool = False,
    ) -> CoverageSnapshot:
        """Advance the simulator and return all vehicles covered during the window."""

        ticks = int(self.config.decision_interval if ticks is None else ticks)
        mrsu_seen = set()
        frsu_seen = set()

        if include_current:
            self._accumulate_coverage_sets(self.mobility.positions(), self.mrsu.position, mrsu_seen, frsu_seen)

        for tick in range(max(0, ticks)):
            if path_plan is not None and tick + 1 < len(getattr(path_plan, "positions", [])):
                self.mrsu.position = float(path_plan.positions[tick + 1]) % float(self.config.road_length)
                if tick < len(getattr(path_plan, "velocities", [])):
                    self.mrsu.speed = float(path_plan.velocities[tick])
            elif not hold_mrsu_position:
                self.mrsu.position = (
                    float(self.mrsu.position) + float(self.mrsu.speed) * float(self.config.dt)
                ) % float(self.config.road_length)
            self.mobility.step(self.config.dt)
            self._accumulate_coverage_sets(self.mobility.positions(), self.mrsu.position, mrsu_seen, frsu_seen)

        return self._coverage_from_sets(mrsu_seen, frsu_seen)

    def _effective_frsu_radius(self) -> float:
        if self.config.frsu_full_coverage:
            return max(float(self.config.frsu_radius), float(self.config.road_length))
        return float(self.config.frsu_radius)

    def _accumulate_coverage_sets(
        self,
        vehicle_positions: Sequence[float],
        mrsu_position: float,
        mrsu_seen: set,
        frsu_seen: set,
    ) -> None:
        snapshot = CoverageModel.snapshot(
            vehicle_positions=vehicle_positions,
            mrsu_position=float(mrsu_position) % float(self.config.road_length),
            mrsu_radius=float(self.config.mrsu_radius),
            frsu_position=float(self.frsu.position),
            frsu_radius=self._effective_frsu_radius(),
            road_length=float(self.config.road_length),
        )
        mrsu_seen.update(int(vehicle_id) for vehicle_id in snapshot.mrsu_covered)
        frsu_seen.update(int(vehicle_id) for vehicle_id in snapshot.frsu_covered)

    def _coverage_from_sets(self, mrsu_seen: set, frsu_seen: set) -> CoverageSnapshot:
        mrsu_covered = sorted(int(vehicle_id) for vehicle_id in mrsu_seen)
        frsu_covered = sorted(int(vehicle_id) for vehicle_id in frsu_seen)
        overlap = sorted(set(mrsu_covered).intersection(frsu_covered))
        return CoverageSnapshot(
            mrsu_covered=mrsu_covered,
            frsu_covered=frsu_covered,
            overlap=overlap,
        )

    def demand_for_vehicles(
        self,
        vehicle_ids: List[int],
        vehicle_requests: Dict[int, List[int]],
    ) -> Counter:
        demand = Counter()
        for vehicle_id in vehicle_ids:
            demand.update(int(content_id) for content_id in vehicle_requests.get(vehicle_id, []))
        return demand

    def content_features(
        self,
        mrsu_vehicle_ids: List[int],
        frsu_vehicle_ids: List[int],
        vehicle_requests: Dict[int, List[int]],
        limit: int,
    ) -> List[dict]:
        mrsu_set = set(mrsu_vehicle_ids)
        frsu_set = set(frsu_vehicle_ids)
        overlap_set = mrsu_set.intersection(frsu_set)
        mrsu_only_ids = sorted(mrsu_set - frsu_set)
        frsu_only_ids = sorted(frsu_set - mrsu_set)
        overlap_ids = sorted(overlap_set)

        mrsu_demand = self.demand_for_vehicles(mrsu_vehicle_ids, vehicle_requests)
        frsu_demand = self.demand_for_vehicles(frsu_vehicle_ids, vehicle_requests)
        mrsu_only_demand = self.demand_for_vehicles(mrsu_only_ids, vehicle_requests)
        frsu_only_demand = self.demand_for_vehicles(frsu_only_ids, vehicle_requests)
        overlap_demand = self.demand_for_vehicles(overlap_ids, vehicle_requests)
        candidate_ids = set()
        candidate_ids.update(content for content, _ in mrsu_demand.most_common(limit))
        candidate_ids.update(content for content, _ in frsu_demand.most_common(limit))
        candidate_ids.update(content for content, _ in mrsu_only_demand.most_common(limit))
        candidate_ids.update(content for content, _ in frsu_only_demand.most_common(limit))
        candidate_ids.update(content for content, _ in overlap_demand.most_common(limit))
        candidate_ids.update(self.global_top_contents[:limit])

        rows = []
        for content_id in candidate_ids:
            rows.append(
                {
                    "content_id": int(content_id),
                    "global_popularity": int(self.global_popularity.get(content_id, 0)),
                    "mrsu_group_popularity": int(mrsu_demand.get(content_id, 0)),
                    "frsu_group_popularity": int(frsu_demand.get(content_id, 0)),
                    "mrsu_only_popularity": int(mrsu_only_demand.get(content_id, 0)),
                    "frsu_only_popularity": int(frsu_only_demand.get(content_id, 0)),
                    "overlap_popularity": int(overlap_demand.get(content_id, 0)),
                }
            )
        rows.sort(
            key=lambda x: (
                x["mrsu_group_popularity"] + x["frsu_group_popularity"],
                x["global_popularity"],
            ),
            reverse=True,
        )
        return rows[:limit]

    def cache_fit_summary(
        self,
        coverage: CoverageSnapshot,
        vehicle_requests: Dict[int, List[int]],
        limit: int = 20,
    ) -> dict:
        mrsu_demand = self.demand_for_vehicles(coverage.mrsu_covered, vehicle_requests)
        frsu_demand = self.demand_for_vehicles(coverage.frsu_covered, vehicle_requests)
        mrsu_feedback = Counter()
        for vehicle_id in coverage.mrsu_covered:
            mrsu_feedback.update(self.last_round_vehicle_missed_counters.get(int(vehicle_id), Counter()))
        frsu_feedback = Counter()
        for vehicle_id in coverage.frsu_covered:
            frsu_feedback.update(self.last_round_vehicle_missed_counters.get(int(vehicle_id), Counter()))
        mrsu_cache_set = set(int(x) for x in self.mrsu_cache)
        frsu_cache_set = set(int(x) for x in self.frsu_cache)

        mrsu_hits = sum(count for content_id, count in mrsu_demand.items() if int(content_id) in mrsu_cache_set)
        frsu_hits = sum(count for content_id, count in frsu_demand.items() if int(content_id) in frsu_cache_set)
        mrsu_missing = [
            {"content_id": int(content_id), "miss_count": int(count)}
            for content_id, count in mrsu_demand.most_common()
            if int(content_id) not in mrsu_cache_set
        ][:limit]
        frsu_missing = [
            {"content_id": int(content_id), "miss_count": int(count)}
            for content_id, count in frsu_demand.most_common()
            if int(content_id) not in frsu_cache_set
        ][:limit]
        mrsu_feedback_contents = [
            {"content_id": int(content_id), "miss_count": int(count)}
            for content_id, count in mrsu_feedback.most_common(limit)
        ]
        frsu_feedback_contents = [
            {"content_id": int(content_id), "miss_count": int(count)}
            for content_id, count in frsu_feedback.most_common(limit)
        ]
        return {
            "mrsu_current_cache_estimated_hits": int(mrsu_hits),
            "frsu_current_cache_estimated_hits": int(frsu_hits),
            "mrsu_top_missing_contents": mrsu_missing,
            "frsu_top_missing_contents": frsu_missing,
            "mrsu_covered_vehicle_feedback_contents": mrsu_feedback_contents,
            "frsu_covered_vehicle_feedback_contents": frsu_feedback_contents,
        }

    def build_agent_context(
        self,
        round_index: int,
        vehicle_requests: Dict[int, List[int]],
        candidate_hotspots: List[dict],
        selected_hotspot: dict = None,
        coverage: CoverageSnapshot = None,
        request_source: str = "provided_request_signal",
    ) -> dict:
        coverage = coverage or self.coverage_snapshot()
        content_features = self.content_features(
            coverage.mrsu_covered,
            coverage.frsu_covered,
            vehicle_requests,
            self.config.global_topk_for_prompt,
        )
        request_counts = {vid: len(reqs) for vid, reqs in vehicle_requests.items()}
        vehicles = [
            {
                "vehicle_id": vehicle.vehicle_id,
                "user_id": vehicle.user_id,
                "position": round(vehicle.position, 2),
                "speed": round(vehicle.speed, 2),
                "predicted_request_count": request_counts.get(vehicle.vehicle_id, 0),
            }
            for vehicle in self.mobility.vehicles
        ]
        enriched_hotspots = [
            self._with_hotspot_mobility_hint(hotspot)
            for hotspot in candidate_hotspots
        ]
        return {
            "round": round_index,
            "request_source": request_source,
            "request_signal_note": (
                "Demand fields are decision-time prediction signals from MovieLens training history "
                "and previous-round feedback. Current-round real requests are hidden until evaluation."
            ),
            "vehicles": vehicles,
            "mrsu_position": round(self.mrsu.position, 2),
            "mrsu_speed": round(self.mrsu.speed, 2),
            "mrsu_motion_model": {
                "road_topology": "circular_one_way",
                "dt": self.config.dt,
                "decision_interval_ticks": int(self.config.decision_interval),
                "v_min": self.config.mrsu_v_min,
                "v_max": self.config.mrsu_v_max,
                "a_min": self.config.mrsu_a_min,
                "a_max": self.config.mrsu_a_max,
                "position_update": "x_next = (x_current + v_current * dt) mod road_length",
                "coverage_distance": "min(|x-y|, road_length-|x-y|)",
                "movement_distance_to_hotspot": "forward circular distance along the fixed travel direction",
                "execution_rule": (
                    "One decision is held for decision_interval_ticks physical steps. "
                    "CHR uses vehicles covered at least once in this service window."
                ),
            },
            "frsu_position": round(self.frsu.position, 2),
            "frsu_full_coverage": bool(self.config.frsu_full_coverage),
            "mrsu_cache": self.mrsu_cache,
            "frsu_cache": self.frsu_cache,
            "candidate_hotspots": enriched_hotspots,
            "selected_hotspot": selected_hotspot,
            "mrsu_covered_vehicles": coverage.mrsu_covered,
            "frsu_covered_vehicles": coverage.frsu_covered,
            "overlap_vehicles": coverage.overlap,
            "candidate_contents": content_features,
            "mrsu_cache_capacity": self.config.mrsu_cache_capacity,
            "frsu_cache_capacity": self.config.frsu_cache_capacity,
            "last_round_hit_ratio": self.last_metrics.chr,
            "last_round_missed_contents": [
                {"content_id": int(content_id), "miss_count": int(count)}
                for content_id, count in self.last_round_missed_counter.most_common(30)
            ],
            "last_round_vehicle_missed_contents": {
                int(vehicle_id): [
                    {"content_id": int(content_id), "miss_count": int(count)}
                    for content_id, count in counter.most_common(8)
                ]
                for vehicle_id, counter in sorted(self.last_round_vehicle_missed_counters.items())
                if counter
            },
            "cache_fit_summary": self.cache_fit_summary(coverage, vehicle_requests),
            "miss_reason_summary": {
                "mbs_miss_count": self.last_metrics.mbs_miss_count,
                "not_covered_count": self.last_metrics.not_covered_count,
                "not_cached_count": self.last_metrics.not_cached_count,
            },
            "user_history_sample": {
                uid: self.user_history.get(uid, [])[:12]
                for uid in self.user_ids[: min(10, len(self.user_ids))]
            },
        }

    def build_tool_decision_context(
        self,
        round_index: int,
        vehicle_requests: Dict[int, List[int]],
        candidate_hotspots: List[dict],
        selected_hotspot: dict = None,
        coverage: CoverageSnapshot = None,
        request_source: str = "provided_request_signal",
        cache_fit_analysis: dict = None,
        fit_summary: dict = None,
        content_features: List[dict] = None,
        dqn_policy_suggestion: dict = None,
    ) -> dict:
        """Build a compact LLM context for tool decisions.

        Full per-content tables are still used by cache tools, but the LLM only
        needs summarized evidence for hotspot and update-flag decisions.
        """

        coverage = coverage or self.coverage_snapshot()
        fit_summary = fit_summary or self.cache_fit_summary(coverage, vehicle_requests)
        content_features = content_features or self.content_features(
            coverage.mrsu_covered,
            coverage.frsu_covered,
            vehicle_requests,
            min(20, self.config.global_topk_for_prompt),
        )
        request_counts = {int(vid): len(reqs) for vid, reqs in vehicle_requests.items()}
        request_total = int(sum(request_counts.values()))
        vehicle_rows = []
        for vehicle in self.mobility.vehicles:
            vehicle_rows.append(
                {
                    "vehicle_id": int(vehicle.vehicle_id),
                    "position": round(float(vehicle.position), 2),
                    "speed": round(float(vehicle.speed), 2),
                    "predicted_request_count": int(request_counts.get(int(vehicle.vehicle_id), 0)),
                }
            )
        top_request_vehicles = sorted(
            vehicle_rows,
            key=lambda item: (int(item["predicted_request_count"]), -int(item["vehicle_id"])),
            reverse=True,
        )[:10]

        context = {
            "context_schema": "compact_tool_decision_v1",
            "round": int(round_index),
            "request_source": request_source,
            "decision_contract": {
                "llm_outputs": [
                    "selected_hotspot_id",
                    "update_mrsu_cache",
                    "update_frsu_cache",
                ],
                "path_smoothness": "computed_by_system_auto_rule",
                "cache_content_ids": "selected_by_DemandAwareCooperativeCacheTool_not_by_LLM",
                "real_requests": "hidden_until_evaluation",
            },
            "system_state": {
                "mrsu": {
                    "position": round(float(self.mrsu.position), 2),
                    "speed": round(float(self.mrsu.speed), 2),
                    "coverage_radius": float(self.config.mrsu_radius),
                    "road_topology": "circular_one_way",
                },
                "frsu": {
                    "position": round(float(self.frsu.position), 2),
                    "full_coverage": bool(self.config.frsu_full_coverage),
                    "coverage_radius": float(self.config.frsu_radius),
                },
                "cache_capacity": {
                    "mrsu": int(self.config.mrsu_cache_capacity),
                    "frsu": int(self.config.frsu_cache_capacity),
                },
                "cache_size": {
                    "mrsu": len(self.mrsu_cache),
                    "frsu": len(self.frsu_cache),
                },
                "motion_constraints": {
                    "road_topology": "circular_one_way",
                    "dt": float(self.config.dt),
                    "decision_interval_ticks": int(self.config.decision_interval),
                    "v_min": float(self.config.mrsu_v_min),
                    "v_max": float(self.config.mrsu_v_max),
                    "a_min": float(self.config.mrsu_a_min),
                    "a_max": float(self.config.mrsu_a_max),
                    "position_update": "x_next = (x_current + v_current * dt) mod road_length",
                    "coverage_distance": "shortest circular distance",
                    "movement_distance_to_hotspot": "forward circular distance only; mRSU does not reverse direction",
                    "execution": "execute decision_interval_ticks physical steps before the next decision",
                    "service_window_coverage": "a vehicle is serviceable if covered at least once within the decision window",
                },
                "last_round_hit_ratio": round(float(self.last_metrics.chr), 6),
            },
            "cache_fit_analysis": _compact_cache_fit_analysis(cache_fit_analysis or {}),
            "cache_fit_summary": _compact_cache_fit_summary(fit_summary),
            "feedback_summary": {
                "miss_reason_summary": {
                    "mbs_miss_count": int(self.last_metrics.mbs_miss_count),
                    "not_covered_count": int(self.last_metrics.not_covered_count),
                    "not_cached_count": int(self.last_metrics.not_cached_count),
                },
                "global_missed_contents_top": _counter_items(self.last_round_missed_counter, 10),
                "mrsu_covered_vehicle_feedback_top": _limit_items(
                    fit_summary.get("mrsu_covered_vehicle_feedback_contents", []),
                    10,
                ),
                "frsu_covered_vehicle_feedback_top": _limit_items(
                    fit_summary.get("frsu_covered_vehicle_feedback_contents", []),
                    10,
                ),
            },
            "candidate_hotspots": [
                self._compact_hotspot(hotspot)
                for hotspot in candidate_hotspots
            ],
            "regional_content_summary_top": _compact_content_features(content_features, 15),
            "request_load_summary": {
                "total_predicted_requests": request_total,
                "active_vehicle_count": sum(1 for count in request_counts.values() if int(count) > 0),
                "top_request_vehicles": top_request_vehicles,
            },
            "coverage_for_cache_evidence": {
                "basis": (
                    "projected_service_window_after_moving_toward_reference_hotspot"
                    if selected_hotspot
                    else "current_position"
                ),
                "reference_hotspot": self._compact_hotspot(selected_hotspot) if selected_hotspot else None,
                "mrsu_covered_vehicle_count": len(coverage.mrsu_covered),
                "mrsu_covered_vehicle_ids": [int(x) for x in coverage.mrsu_covered],
                "frsu_covered_vehicle_count": len(coverage.frsu_covered),
                "overlap_vehicle_count": len(coverage.overlap),
            },
        }
        if dqn_policy_suggestion is not None:
            context["dqn_policy_suggestion"] = dqn_policy_suggestion
        return context

    def _with_hotspot_mobility_hint(self, hotspot: dict) -> dict:
        enriched = dict(hotspot)
        position = float(enriched.get("position", 0.0))
        enriched["distance_to_mrsu"] = round(
            CoverageModel.forward_distance(self.mrsu.position, position, self.config.road_length),
            2,
        )
        enriched["forward_distance_to_mrsu"] = enriched["distance_to_mrsu"]
        enriched["circular_coverage_distance_to_mrsu"] = round(
            CoverageModel.circular_distance(self.mrsu.position, position, self.config.road_length),
            2,
        )
        return enriched

    def _compact_hotspot(self, hotspot: dict) -> dict:
        enriched = self._with_hotspot_mobility_hint(hotspot)
        return {
            "hotspot_id": int(enriched.get("hotspot_id", 0)),
            "position": round(float(enriched.get("position", 0.0)), 2),
            "distance_to_mrsu": round(float(enriched.get("distance_to_mrsu", 0.0)), 2),
            "forward_distance_to_mrsu": round(float(enriched.get("forward_distance_to_mrsu", 0.0)), 2),
            "circular_coverage_distance_to_mrsu": round(
                float(enriched.get("circular_coverage_distance_to_mrsu", 0.0)),
                2,
            ),
            "covered_vehicle_count": int(enriched.get("covered_vehicle_count", 0)),
            "covered_vehicle_ids": [int(x) for x in enriched.get("covered_vehicle_ids", [])],
            "potential_cache_gain": round(float(enriched.get("potential_cache_gain", 0.0)), 4),
            "dominant_contents_top": [int(x) for x in enriched.get("dominant_contents", [])[:5]],
            "demand_summary_top": _mapping_items(enriched.get("demand_summary") or {}, 5),
        }

    def apply_path_plan(self, positions: List[float], velocities: List[float]) -> None:
        if len(positions) > 1:
            self.mrsu.position = float(positions[1])
        if velocities:
            self.mrsu.speed = float(velocities[0])

    def set_cache(self, mrsu_cache: List[int], frsu_cache: List[int]) -> None:
        self.mrsu_cache = [int(item) for item in mrsu_cache]
        self.frsu_cache = [int(item) for item in frsu_cache]

    def evaluate(
        self,
        vehicle_requests: Dict[int, List[int]],
        coverage: CoverageSnapshot,
        mrsu_cache: List[int],
        frsu_cache: List[int],
    ) -> RoundMetrics:
        (
            self.last_round_missed_counter,
            self.last_round_vehicle_missed_counters,
        ) = self._missed_content_counters(
            vehicle_requests,
            coverage,
            mrsu_cache,
            frsu_cache,
        )
        metrics = evaluate_cache_hit_ratio(
            vehicle_requests=vehicle_requests,
            mrsu_covered=coverage.mrsu_covered,
            frsu_covered=coverage.frsu_covered,
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
        )
        self.last_metrics = metrics
        return metrics

    def _missed_content_counters(
        self,
        vehicle_requests: Dict[int, List[int]],
        coverage: CoverageSnapshot,
        mrsu_cache: List[int],
        frsu_cache: List[int],
    ) -> Tuple[Counter, Dict[int, Counter]]:
        mrsu_covered_set = set(coverage.mrsu_covered)
        frsu_covered_set = set(coverage.frsu_covered)
        mrsu_cache_set = set(int(x) for x in mrsu_cache)
        frsu_cache_set = set(int(x) for x in frsu_cache)
        missed = Counter()
        missed_by_vehicle: Dict[int, Counter] = defaultdict(Counter)
        for vehicle_id, requests in vehicle_requests.items():
            in_mrsu = vehicle_id in mrsu_covered_set
            in_frsu = vehicle_id in frsu_covered_set
            for content_id in requests:
                content_id = int(content_id)
                if in_mrsu and content_id in mrsu_cache_set:
                    continue
                if in_frsu and content_id in frsu_cache_set:
                    continue
                missed[content_id] += 1
                missed_by_vehicle[int(vehicle_id)][content_id] += 1
        return missed, missed_by_vehicle


def _mapping_items(mapping: dict, limit: int, value_key: str = "predicted_count") -> List[dict]:
    items = [
        (int(key), int(value))
        for key, value in (mapping or {}).items()
        if int(value) > 0
    ]
    items.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return [
        {"content_id": int(content_id), value_key: int(value)}
        for content_id, value in items[: int(limit)]
    ]


def _counter_items(counter: Counter, limit: int, value_key: str = "miss_count") -> List[dict]:
    return [
        {"content_id": int(content_id), value_key: int(count)}
        for content_id, count in Counter(counter or {}).most_common(int(limit))
    ]


def _limit_items(items: List[dict], limit: int) -> List[dict]:
    compact = []
    for item in (items or [])[: int(limit)]:
        row = {}
        for key, value in item.items():
            if isinstance(value, (int, np.integer)):
                row[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                row[key] = round(float(value), 6)
            else:
                row[key] = value
        compact.append(row)
    return compact


def _compact_cache_fit_summary(fit_summary: dict, limit: int = 10) -> dict:
    fit_summary = fit_summary or {}
    return {
        "mrsu_current_cache_estimated_hits": int(fit_summary.get("mrsu_current_cache_estimated_hits", 0)),
        "frsu_current_cache_estimated_hits": int(fit_summary.get("frsu_current_cache_estimated_hits", 0)),
        "mrsu_top_missing_contents": _limit_items(
            fit_summary.get("mrsu_top_missing_contents", []),
            limit,
        ),
        "frsu_top_missing_contents": _limit_items(
            fit_summary.get("frsu_top_missing_contents", []),
            limit,
        ),
    }


def _compact_cache_fit_analysis(analysis: dict, candidate_limit: int = 10) -> dict:
    analysis = analysis or {}
    keys = [
        "estimated_keep_chr",
        "estimated_tool_update_chr",
        "estimated_acr_update_chr",
        "estimated_gain",
        "dominant_content_overlap",
        "dominant_content_count",
        "dominant_content_overlap_ratio",
        "keep_mrsu_hit",
        "keep_frsu_hit",
        "tool_update_mrsu_hit",
        "tool_update_frsu_hit",
        "acr_update_mrsu_hit",
        "acr_update_frsu_hit",
        "cache_update_tool",
    ]
    compact = {}
    for key in keys:
        if key not in analysis:
            continue
        value = analysis[key]
        if isinstance(value, (int, np.integer)):
            compact[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            compact[key] = round(float(value), 6)
        else:
            compact[key] = value
    compact["mrsu_candidate_head"] = [
        int(x) for x in analysis.get("mrsu_candidate_head", [])[: int(candidate_limit)]
    ]
    compact["frsu_candidate_head"] = [
        int(x) for x in analysis.get("frsu_candidate_head", [])[: int(candidate_limit)]
    ]
    return compact


def _compact_content_features(content_features: List[dict], limit: int = 15) -> List[dict]:
    rows = []
    for item in (content_features or [])[: int(limit)]:
        rows.append(
            {
                "content_id": int(item.get("content_id", 0)),
                "global_popularity": int(item.get("global_popularity", 0)),
                "mrsu_group_popularity": int(item.get("mrsu_group_popularity", 0)),
                "frsu_group_popularity": int(item.get("frsu_group_popularity", 0)),
                "mrsu_only_popularity": int(item.get("mrsu_only_popularity", 0)),
                "frsu_only_popularity": int(item.get("frsu_only_popularity", 0)),
                "overlap_popularity": int(item.get("overlap_popularity", 0)),
            }
        )
    return rows


def _copy_counter_dict(source: Dict[int, Counter]) -> Dict[int, Counter]:
    return {int(key): Counter(value) for key, value in source.items()}


def _integer_allocations(expected_counts: Dict[int, float], target: int) -> Dict[int, int]:
    if target <= 0 or not expected_counts:
        return {int(vehicle_id): 0 for vehicle_id in expected_counts}

    floored = {
        int(vehicle_id): int(np.floor(max(0.0, float(value))))
        for vehicle_id, value in expected_counts.items()
    }
    allocated = sum(floored.values())
    remainder = int(target) - allocated
    fractions = sorted(
        expected_counts.items(),
        key=lambda item: (float(item[1]) - np.floor(max(0.0, float(item[1]))), -int(item[0])),
        reverse=True,
    )

    if remainder > 0:
        for idx in range(remainder):
            vehicle_id = int(fractions[idx % len(fractions)][0])
            floored[vehicle_id] += 1
    elif remainder < 0:
        for vehicle_id, _ in sorted(floored.items(), key=lambda item: item[1], reverse=True):
            if remainder == 0:
                break
            removable = min(floored[vehicle_id], -remainder)
            floored[vehicle_id] -= removable
            remainder += removable
    return floored


def load_preprocessed_movielens(data_dir: str = "datasets", prefix: str = "ml1m") -> Tuple[list, list]:
    """Load the same preprocessed MovieLens pickle artifacts used by the legacy project.

    This intentionally avoids importing dataset_processing at module import time because
    that legacy module imports plotting/dataframe packages eagerly.
    """

    data_path = Path(data_dir)
    with open(data_path / f"{prefix}_train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(data_path / f"{prefix}_test.pkl", "rb") as f:
        test = pickle.load(f)
    return train, test


def _normalize_row(row) -> Tuple[int, int, int, float, int, int]:
    return (
        int(row[0]),
        int(row[1]),
        int(row[2]),
        float(row[3]),
        int(row[4]),
        int(row[5]),
    )


def _first_unique_users(rows: List[Tuple[int, int, int, float, int, int]], limit: int) -> List[int]:
    users = []
    seen = set()
    for row in rows:
        user_id = int(row[0])
        if user_id in seen:
            continue
        seen.add(user_id)
        users.append(user_id)
        if len(users) >= limit:
            break
    return users
