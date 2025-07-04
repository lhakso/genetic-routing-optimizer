# Vehicle Routing Optimization: Experimentation & Development Log

This document outlines the iterative development, experimentation, and performance optimization efforts undertaken during the creation of a memetic algorithm for solving large-scale vehicle routing problems (VRP) in Seattle. It provides a transparent record of algorithmic evolution, performance tuning, and key learnings throughout the project lifecycle.

---

## Phase 1: Data Setup and Initial Testing

* Set up a Linux-based OSRM server for distance matrix computation.
* Cleaned raw job site data with a focus on modular, loggable preprocessing (May 13).
* Generated coordinates and distance matrix for test cases (May 15).
* Deferred handling of irregular data until the algorithm phase was functional.

## Phase 2: Baseline Genetic Algorithm Trials

* Implemented a basic genetic algorithm with mutation-only logic.
* Observed severe local minima trapping — best route score: \~4,190,000 meters.
* Introduced parent crossover alongside mutation (May 16).

  * Minor improvement; best score reduced to \~3,970,000 meters.

## Phase 3: Population Scaling and Performance Bottlenecks

* Increased population size to 100.

  * Resulted in a substantial improvement: \~814,000 meters over 50,000 generations.
* Compute became a bottleneck with large generation counts.
* Ported fitness evaluation and distance matrix logic to Rust for performance.

  * Achieved a **22.5x speedup**:

    * Python (no initializer): 820.40 seconds for 5,000 generations
    * Rust: 36.38 seconds for 5,000 generations
  * Best Rust-accelerated score: 812,240.90 meters

## Phase 4: Genetic Strategy Refinement

* Experimented with:

  * Increased population (200)
  * Lower generation counts (10,000)
  * Adjusted parent selection count
  * Best score plateaued around 433,583.30 meters

* Zoomed in on dense regions of the map to target bottlenecks.

  * Population: 500, Parent Selection Count: 20, 10,000 generations
  * Plateaued near 8,100 generations, score: 371,022.60 meters

## Phase 5: Selection Strategy Experiments

* Tested tournament selection — results degraded by \~25%
* Introduced hybrid selection:

  * First 5,000 generations with standard selector
  * Then switched to tournament (split population into equal parts, select top-N)
  * Score: 339,428.80 meters
* Pure truncation selection produced similar results (338,748.00 meters) but lost diversity faster

## Phase 6: Local Search Integration

* Added 2-opt improvements

  * Initial runs led to local minima at \~263,000 meters
  * Introduced 5-pass 2-opt at random start points after 1,000 generations
  * Result: 261,000 meters

* Further refinement:

  * Full 2-opt passes on elites, random passes on children
  * Improved to 241,853.40 meters
  * Increased parent selection count, aggressive 2-opt: \~236,000 meters

## Phase 7: Clustering Strategy

* Switched to a cluster-first routing approach using hierarchical clustering

  * Computed second derivative to find optimal cut point
  * Focused testing on a cluster with 534 job sites

## Phase 8: High-Precision Optimization

* Achieved best route score: 65,177 meters

  * Relied heavily on tuned 2-opt and 3-opt routines

* Key finding:

  * Performing fewer 2-opt passes before 3-opt significantly reduced compute time
  * Route quality remained nearly identical, implying diminishing returns from extensive 2-opt

## Tools & Technologies Used

* Python (Genetic Algorithm Core)
* Rust (Fitness Evaluation and Local Search)
* OSRM (Open Source Routing Machine)
* NumPy, Pandas
* PyO3 (Rust-Python FFI)
* Matplotlib/Folium (Route visualization)

## Ongoing and Future Directions

* Testing additional metaheuristics (e.g., simulated annealing)
* Expanding cluster routing to multi-depot problems
* Experimenting with machine learning–guided heuristics for route initialization
* Evaluating constraint-aware scoring with soft penalties

---

*This document supports the development narrative behind the Vehicle Routing Problem Solver and complements the main README. It captures the iterative engineering and decision-making process behind performance breakthroughs and algorithmic improvements.*
