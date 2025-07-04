# Vehicle Routing Problem Solver: Seattle Crew Optimization

A genetic algorithm implementation for optimizing complex routing across 800+ job sites in Seattle. This project demonstrates advanced algorithm design, performance optimization, and practical application of computational intelligence to real-world logistics challenges.

### Research Project Disclaimer

> This project is a **research-style, exploratory system** developed to test hybrid optimization strategies and algorithmic performance for large-scale vehicle routing. It is **not a production-ready system**, but rather an advanced prototype for experimenting with metaheuristics and performance engineering.  
>  
> I have experience working on **production-grade software systems**, but this project prioritizes **algorithmic exploration and speed** over deployment, fault tolerance, or user-facing robustness.


## **Project Overview**

This project implements a custom memetic algorithm that combines genetic algorithms with local search optimization to solve large-scale Vehicle Routing Problems (VRP). The system was designed to optimize crew routing across Seattle's electrical infrastructure, processing hundreds of job sites with complex constraints including equipment requirements, working hours, and geographical limitations.

**Key Achievement**: Developed a hybrid algorithm combining genetic algorithms with Rust-optimized local search, demonstrating systematic improvements through algorithmic innovation and performance optimization.

## **Technical Highlights**

### Core Technologies
- **Python 3.11+**: Primary implementation language
- **Rust**: High-performance fitness evaluation and local search optimization
- **PyO3**: Seamless Python-Rust interoperability
- **NumPy**: Efficient numerical computations and distance matrix operations
- **Pandas**: Data processing and analysis
- **Folium**: Interactive route visualization and mapping

### Algorithm Architecture
- **Memetic Algorithm**: Hybrid approach combining genetic algorithms with local search
- **Multi-threaded Processing**: Leverages multiprocessing for population evaluation
- **Adaptive Selection**: Tournament and elite selection strategies
- **Advanced Crossover**: Order crossover (OX) and Edge Recombination (ERX)
- **Local Search**: 2-opt and 3-opt optimization implemented in Rust

### Performance Optimizations
- **Rust Integration**: Fitness evaluation ported to Rust for ~10x speed improvement
- **Parallel Processing**: Multi-core utilization for population-based operations
- **Memory Management**: Efficient distance matrix handling for large datasets
- **Caching Strategy**: Optimized data structures for repeated calculations

## **Architecture & Design**

### Project Structure
```
src/
├── ga/                          # Genetic Algorithm Core
│   ├── genetic_router.py        # Main GA implementation
│   ├── gene_selectors.py        # Selection strategies
│   ├── crossover.py             # Crossover operators
│   ├── ga_helpers.py            # Utility functions
│   └── visualizers.py           # Route visualization
├── ga_helpers_rs/               # Rust Performance Module
│   ├── src/lib.rs               # Core Rust implementations
│   └── Cargo.toml               # Rust dependencies
├── auth_utils.py                # Google Sheets integration
├── clean_sheet.py               # Data processing utilities
└── main.py                      # Entry point
```

### Key Components

**1. Genetic Algorithm Engine** (`genetic_router.py`)
- Population initialization and management
- Multi-generational evolution with adaptive parameters
- Integrated performance monitoring and logging
- Support for various selection and crossover strategies

**2. Rust Performance Module** (`ga_helpers_rs/`)
- High-performance fitness evaluation
- 2-opt and 3-opt local search implementations
- Memory-efficient distance matrix operations
- FFI bindings for seamless Python integration

**3. Data Processing Pipeline** (`clean_sheet.py`, `auth_utils.py`)
- Google Sheets API integration for real-time data access
- Address geocoding and coordinate processing
- Distance matrix computation and optimization
- Data validation and cleaning procedures

## **Installation & Setup**

### Prerequisites
- Python 3.13+
- Rust 1.70+
- Google Sheets API credentials (for data access)

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/vehicle-routing-optimizer.git
cd vehicle-routing-optimizer
```

2. **Set Up Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Build Rust Components**
```bash
cd src/ga_helpers_rs
cargo build --release
cd ../..
```

4. **Configure Authentication**
```bash
# Create config directory
mkdir config

# Add your Google Sheets credentials
# Place credentials.json in config/ directory
# Note: Never commit credential files to version control
```

### Dependencies
```python
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
folium>=0.14.0
gspread>=5.0.0
requests>=2.31.0

# Development dependencies
pytest>=7.0.0
black>=23.0.0
mypy>=1.0.0
```

## **Usage**

### Basic Usage
```python
from src.ga.genetic_router import find_route_mp
from src.ga.visualizers import visualize_route
import numpy as np
import pandas as pd

# Load your coordinate data
coords_df = pd.read_csv('coordinates.csv')
coords_list = list(zip(coords_df['lat'], coords_df['lng']))

# Load distance matrix
distance_matrix = np.load('distance_matrix.npy')

# Configure GA parameters
GA_PARAMS = {
    'population_size': 500,
    'num_generations': 1000,
    'elite_count': 5,
    'mutation_prob': 0.05,
    'parent_selection_count': 100
}

# Run optimization
best_score, best_route, fitness_history, generations = find_route_mp(
    coords_list,
    distance_matrix,
    **GA_PARAMS
)

# Visualize results
route_map = visualize_route(best_route, coords_list)
route_map.save('optimized_route.html')
```

### Advanced Configuration
```python
# Custom fitness function with constraints
def custom_fitness(route, matrix, constraints=None):
    base_distance = calculate_route_distance(route, matrix)
    
    # Add constraint penalties
    if constraints:
        penalty = apply_constraints(route, constraints)
        return base_distance + penalty
    
    return base_distance

# Multi-objective optimization
MULTI_OBJECTIVE_PARAMS = {
    'objectives': ['distance', 'time', 'fuel_cost'],
    'weights': [0.4, 0.4, 0.2],
    'constraint_penalties': True
}
```

## **Performance Metrics**

### Benchmarking Results
- **Scale**: Successfully optimized routes for 800+ job sites
- **Performance**: Significant speed improvement through Rust integration
- **Quality**: Demonstrated systematic improvement over baseline routes
- **Scalability**: Efficient scaling with problem size through parallel processing

### Algorithm Performance
The system demonstrates measurable improvements through:
- **Rust Integration**: Substantial performance gains in fitness evaluation
- **Parallel Processing**: Multi-core utilization for population-based operations
- **Memory Optimization**: Efficient distance matrix handling for large datasets
- **Algorithm Refinement**: Iterative improvements shown in performance plots

## **Research & Development**

### Algorithm Innovations
1. **Adaptive Selection Pressure**: Dynamic tournament size based on population diversity
2. **Hybrid Local Search**: Combining 2-opt and 3-opt with problem-specific heuristics
3. **Memory-Efficient Representation**: Optimized data structures for large-scale problems
4. **Parallel Fitness Evaluation**: Lock-free concurrent processing

### Future Experimental Features
- **Simulated Annealing Integration**: Hybrid metaheuristic approaches
- **Machine Learning Integration**: Learned heuristics for initialization
- **Real-time Optimization**: Dynamic re-routing based on traffic conditions
- **Multi-depot Extensions**: Support for multiple starting locations

## **Contributing**

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with performance benchmarks

## **Technical Documentation**

### Algorithm Details
- **Genetic Representation**: Permutation encoding for route sequences
- **Selection Strategy**: Tournament selection with adaptive pressure
- **Crossover Operators**: Order crossover (OX) and Edge Recombination (ERX)
- **Mutation**: Swap mutation with adaptive rates
- **Local Search**: 2-opt and 3-opt improvements

### Performance Analysis
- **Time Complexity**: O(n² × p × g) where n=sites, p=population, g=generations
- **Space Complexity**: O(n² + p×n) for distance matrix and population storage
- **Scalability**: Linear scaling with multiprocessing optimization

## **Key Achievements**

- **Scalability**: Successfully handles 800+ job sites
- **Performance**: Significant speed improvement through Rust integration
- **Quality**: Demonstrated systematic improvement over baseline approaches
- **Real-world Relevance**: Evaluated against human-planned routes; an earlier prototype was field-tested with Seattle-based utility crews
- **Technical Innovation**: Novel hybrid algorithm design

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Connect**

- **LinkedIn**: https://www.linkedin.com/in/luke-hakso-a4a672330/
- **Email**: luke.c.hakso@gmail.com

---

*The system showcases how modern hybrid algorithms and language interoperability (Python + Rust) can tackle combinatorial logistics problems at scale.* 
