# Mind the Graph

## The Role of Topology in Quantum Annealing

Quantum annealing offers a promising framework for addressing NP-hard optimization problems, yet its effectiveness is fundamentally constrained by hardware topology.
Mapping a problem into a quantum processing unit (QPU) requires minor embedding, where logical variables are represented by chains of physical qubits.
This step introduces both overhead and error sensitivity, making the study of QPU connectivity a critical aspect of performance.

In this work we investigate the impact of topology on embeddability by comparing Zephyr graphs, used in current D-Wave devices, with alternative Havel-Hakimi constructions.
We focus on the ability of these topologies to host large cliques, a canonical case for dense optimization problems, and evaluate embedding performance over a wide parameter space.

Our findings reveal structural factors that strongly influence embedding capacity, providing insights relevant to QPU design.

## Structure of the repository

`data_generator.py`: main file to create the dataset, in the function `dispatch_qpu` you can add your custom topology.

`report`: TeX sources of the paper
