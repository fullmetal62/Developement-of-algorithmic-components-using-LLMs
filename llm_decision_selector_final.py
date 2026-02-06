# inside llm_decision_selector.py

import time
import logging
from utilslogger import log_iteration
from my_mutation1 import MyExtendedMutation1
from my_mutation2 import MyExtendedMutation2
from BaseClasses import (
    RandomHillclimber,
    Repeat,
    RuinAndRecreate,
    VariableNeighbourhoodSearch,
    TabuSearchComponent,
    HybridGeneticComponent,
    AdaptiveOperatorSelection,
)
from llm_prompting2 import ask_llm_for_component_name
import random
from neighbourhood import *
import os
from dictionaries import AVAILABLE_COMPONENTS, MUTATION_COMPONENTS





class LLMDrivenAlgorithm:
    def __init__(self, solution_class, instance_data, max_iterations: int | None = None):
        self.solution_class = solution_class
        self.instance_data = instance_data
        self.available_components = AVAILABLE_COMPONENTS
        self.mutation_components= MUTATION_COMPONENTS
        self.max_iterations = max_iterations
        self.controllers={"Repeat", "AdaptiveOperatorSelection"}
        self.logger = logging.getLogger("tsp_llm.algorithm")

    def build_component(self, component_name: str, llm_args: dict, instance):
  
        ComponentClass = self.available_components[component_name]

        if component_name == "RandomHillClimber":
            mutation_name = llm_args.get("mutation_component", "MyExtendedMutation1")
            mutation_cls = MyExtendedMutation1 if mutation_name == "MyExtendedMutation1" else MyExtendedMutation2
            iterations = llm_args.get("iterations", 1000)
            return ComponentClass(mutation_cls(), iterations)

        if component_name =="Repeat":
            
            repeat_iters = int(llm_args.get("iterations", random.randint(2, 8)))
            inner_name = llm_args.get("inner_component")
            inner_args = llm_args.get("inner_args", {})  

            #updating inner component list
            inner_candidates = {k: v for k, v in self.available_components.items()
                                if k not in self.controllers}
            inner_candidates.update({"Swap 2 cities": MyExtendedMutation1, "Reverse-2opt": MyExtendedMutation2})

            # 3) If outer args didn’t specify a valid inner, ask LLM ONCE for the inner
            if not inner_name or inner_name not in inner_candidates:
                current_solution= self.solution_class(instance)
                inner_name, inner_args2 = ask_llm_for_component_name(
                    current_solution, instance, inner_candidates
                )
                # prefer outer-provided inner_args if present; otherwise use returned ones
                if not inner_args:
                    inner_args = inner_args2 or {}

            # 4) Build the inner component INSTANCE with sensible defaults
            if inner_name in self.mutation_components:
                return Repeat(self.mutation_components[inner_name], repeat_iters)
            inner_component = self.build_component(inner_name, inner_args, instance)

            # 5) Wrap with Repeat and return
            return Repeat(inner_component, repeat_iters)

        if component_name == "RuinAndRecreate":
            return ComponentClass(instance)

        if component_name == "VariableNeighbourhoodSearch":
            
            neighbours = [TwoOptNeighbourhood(), SwapNeighbourhood()]
            shake = int(llm_args.get("shake_strength", 1))       # allow LLM to tweak
            loops = int(llm_args.get("max_outer_loops", 10))     # small, bounded
            component = VariableNeighbourhoodSearch(
                neighbourhoods=neighbours,
                kmax=len(neighbours),
                shake_strength=shake,
                max_outer_loops=loops
            )
            return component
        if component_name == "TabuSearchComponent":
            mutation = MyExtendedMutation1()   or MyExtendedMutation2()
            iterations = llm_args.get("iterations", 2500)
            tabu_tenure = llm_args.get("tabu_tenure", 10)
            return ComponentClass(mutation, iterations, tabu_tenure)

        if component_name == "HybridGeneticComponent":
            mut =  llm_args.get("mutation_component", "MyExtendedMutation1")
            mutation_cls = MyExtendedMutation1 if mut == "MyExtendedMutation1" else MyExtendedMutation2
            hgc = ComponentClass(self.solution_class, mutation_cls())
            hgc.initialize_population(self.instance_data)
            return hgc

        if component_name == "AdaptiveOperatorSelection":
            controllers = {"Repeat", "AdaptiveOperatorSelection"}
            candidates = {k: v for k, v in self.available_components.items() if k not in controllers}
            if not hasattr(self, "aos_component"):
                # Pass the whole components dict; AOS will choose among the names
                aos = AdaptiveOperatorSelection(
                    components_dict=candidates,
                    solution_class=self.solution_class,
                    instance_data=instance,
                )

                log_path = "llm_prediction_log.csv"
                if os.path.exists(log_path) and hasattr(aos, "bootstrap_from_history"):
                    try:
                        aos.bootstrap_from_history(log_path)
                        print("[AOS Bootstrapped] Loaded operator success data from log.")
                    except Exception as e:
                        print("[AOS Bootstrapping Failed]", e)

                self.aos_component = aos

            return self.aos_component
        raise ValueError(f"Unknown component: {component_name}")

    def solve(self, instance, time_budget_ms: int, logger: logging.Logger | None = None):
        if logger is not None:
            self.logger = logger

        time_budget_ns = time_budget_ms * 1e6
        start_ns = time.time_ns()

        current_solution = self.solution_class(instance)
        best_solution = current_solution.clone()
        best_cost = best_solution.get_objective()

        iteration = 0

        while time.time_ns() - start_ns < time_budget_ns:
            iteration += 1
            if self.max_iterations is not None and iteration > self.max_iterations:
                break

            # 1) Get a component decision from LLM
            component_name, llm_args =ask_llm_for_component_name(
                current_solution,  self.available_components
            )

            # 2) Build the actual component object
            try:
                component = self.build_component(component_name, llm_args, instance)
            except Exception as e:
                self.logger.warning(
                    "Failed to build component '%s' from args %r (%s); skipping iteration.",
                    component_name,
                    llm_args,
                    e,
                )
                continue

            # 3) Apply component and time it
            before_cost = current_solution.get_objective()
            before_tour = list(current_solution.solution)

            start_iter = time.perf_counter()
            try:
                new_solution = component.apply(current_solution)
            except Exception as e:
                self.logger.warning(
                    "Error applying component %s at iteration %d: %s",
                    component_name,
                    iteration,
                    e,
                )
                continue
            duration_ms = (time.perf_counter() - start_iter) * 1000.0

            after_cost = new_solution.get_objective()
            after_tour = list(new_solution.solution)
            improved = after_cost < before_cost

            # 4) Logging
            self.logger.info(
                "iter=%d comp=%s before=%.3f after=%.3f delta=%.3f time_ms=%.1f",
                iteration,
                component_name,
                before_cost,
                after_cost,
                after_cost - before_cost,
                duration_ms,
            )

            log_iteration(
                iteration=iteration,
                component_name=component_name,
                llm_selected=True,
                llm_args=llm_args,
                success=improved,
                old_cost=before_cost,
                new_cost=after_cost,
                duration_ms=duration_ms,
                tour_before=before_tour,
                tour_after=after_tour,
            )

            # 5) Update current and best
            current_solution = new_solution
            if after_cost < best_cost:
                best_solution = new_solution.clone()
                best_cost = after_cost

        return best_solution
