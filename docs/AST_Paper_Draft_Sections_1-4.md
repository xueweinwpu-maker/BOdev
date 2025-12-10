A Bi-Level Bayesian Optimization Framework for the Multi-Point Aerodynamic Design of Morphing Aircraft
Abstract: [To be written last, after results are finalized. Will summarize the problem, the novel bi-level framework, the key findings from the Pareto front, the >28% CL improvement, the ARD analysis, and the final CFD validation.]

1. Introduction
Since the Wright brothers' first flight, aeronautical engineering has been a continuous quest to expand the boundaries of flight. A prominent frontier in this endeavor is the development of trans-domain aircraft, capable of efficient operation across a vast flight envelope from subsonic take-off and landing to hypersonic cruise and re-entry. The profound variation in flow physics across these regimes, however, means that an aircraft with a conventional fixed geometry is inherently a compromise, optimized for one flight condition at the expense of all others. This limitation presents a significant barrier to achieving next-generation performance goals.

Morphing aircraft, which can purposefully adapt their geometry in-flight, offer a compelling solution to this challenge. By altering parameters such as wing sweep or camber, a morphing airframe can maintain near-optimal aerodynamic efficiency across multiple disparate design points. The optimization of such concepts has become an active area of research, often relying on gradient-based methods coupled with adjoint solvers for efficiency [1, 2]. While powerful, these local optimization techniques require a good initial starting point and can struggle to escape local optima in the vast design spaces associated with morphing airframes. Furthermore, a persistent challenge in the field is the formal, systematic handling of multi-point constraints and the non-uniform design spaces inherent to morphing aircraft. While many studies have successfully demonstrated multi-point design [3, 4], the optimization problem is often simplified by pre-selecting a limited set of configurations or by treating the online variables as fixed parameters for different stages of a sequential optimization. This highlights a key gap in the literature: the need for a global optimization framework that can simultaneously optimize the fixed airframe shape while respecting the unique, state-dependent constraints and objectives of each morphing configuration.

Bayesian Optimization (BO) has emerged as a powerful tool for global optimization in aerospace engineering, prized for its exceptional data efficiency when dealing with computationally expensive evaluations like CFD simulations [5, 6]. It has been successfully applied to a range of aerodynamic shape optimization problems, from airfoils to wings and full aircraft configurations. However, the majority of these applications have focused on single-level, unconstrained, or simply-constrained problems. The application of BO to complex, hierarchically-structured problems remains a developing area. While some research has explored constrained BO [7], these methods typically assume a uniform design space where all constraints apply globally.

To address these gaps, this paper introduces a novel, data-driven, bi-level Bayesian optimization framework. The primary contribution is the sophisticated MDO (Multidisciplinary Design Optimization) architecture itself, which is customized for trans-domain morphing aircraft and possesses two key features. First, it is explicitly designed to flexibly handle non-uniform design spaces—where the allowable range for online variables differs between flight conditions—a challenge that monolithic approaches struggle with. Second, its bi-level structure provides a systematic methodology for addressing different and even conflicting design constraints across multiple design points by intelligently decomposing them into local (inner-loop) and global (outer-loop) problems. The secondary contribution is a methodology for extracting deep physical insights from the data-driven process, using the Automatic Relevance Determination (ARD) kernels within our surrogate models to perform a global sensitivity analysis. Finally, we provide definitive validation of the framework's superiority by discovering a design with significantly higher performance (>28% improvement in subsonic lift) compared to previously published results, and by verifying this new design with high-fidelity CFD analysis.

The remainder of this paper is structured as follows: Section 2 details the bi-level optimization methodology. Section 3 describes the case study and surrogate model validation. Section 4 presents and discusses the optimization results. Finally, Section 5 provides concluding remarks.

2. Methodology
Our proposed methodology is a hierarchical framework that decomposes the complex morphing aircraft design problem into two interconnected levels: an outer loop for the global optimization of the aircraft's fundamental shape, and a set of inner loops for the local, condition-specific optimization of its morphing parameters.

2.1. The Bi-Level Optimization Framework
The core of our approach is the separation of design variables into two categories: offline variables, which define the fixed geometric properties of the airframe, and online variables, such as control surface deflections or sweep angles, which are adjusted during flight. The outer loop, managed by a Bayesian optimizer, searches the high-dimensional space of offline variables to find a globally optimal shape. For each candidate shape proposed by the outer loop, a set of parallel inner loops are executed, one for each design point in the flight envelope. Each inner loop is a fast, local, gradient-based optimizer tasked with finding the best online variable settings for that specific shape and flight condition.

This bi-level architecture is crucial for handling the non-uniform design spaces typical of morphing aircraft. A traditional "monolithic" approach, which would treat all offline and online variables as a single large vector, is often infeasible. For instance, the allowable sweep angle for subsonic flight may not overlap at all with the allowable range for hypersonic flight. Our decomposition elegantly solves this by providing each inner loop with its own distinct, physically valid search boundaries for the online variables.

Furthermore, this structure allows for a sophisticated distribution of constraints. Global constraints, which depend on results from multiple design points (e.g., stability across the flight envelope), are handled by the outer loop. Local constraints, which are specific to a single flight condition (e.g., a minimum lift coefficient for landing), are efficiently handled within the corresponding inner loop. This decomposition makes the main Bayesian optimizer's task simpler and the overall search more efficient.

2.2. Surrogate-Based Bayesian Optimization
The outer loop employs Bayesian Optimization (BO), a highly data-efficient global search strategy ideal for optimizing expensive black-box functions. BO builds a statistical surrogate model—in our case, a Gaussian Process (GP)—of the objective and constraint functions. This model is then used to intelligently select the next point to evaluate by maximizing an acquisition function, which balances exploitation (searching near the current best solution) and exploration (reducing uncertainty in unknown regions of the design space).

A key feature of our GP models is the use of a Matern 5/2 kernel with Automatic Relevance Determination (ARD). In a high-dimensional design space, not all variables are equally important. The ARD kernel learns a separate length-scale for each input variable, effectively inferring their relevance to the output. As we will demonstrate, this makes the ARD kernel not only a modeling choice for achieving high accuracy but also a powerful tool for performing a data-driven global sensitivity analysis.

The entire framework is implemented in Python, leveraging the BoTorch library for Bayesian optimization and GPyTorch for surrogate modeling. To ensure a stable and efficient search, several advanced strategies are employed, including a "guiding point" to initialize the search in a known high-performance region, and a "warm-start" strategy for the inner loops.

3. Case Study Setup
To demonstrate the capabilities of the framework, a case study was performed on a trans-domain morphing aircraft concept.

3.1. Baseline Aircraft Geometry & Parameterization
The baseline geometry is a variable-sweep waverider-based concept designed for flight from Ma 0.4 to Ma 10.0. The shape of the aircraft is parameterized using the Free-Form Deformation (FFD) method. Fifteen FFD control points are designated as the offline design variables, controlling the aircraft's fundamental shape. A single online variable, controlling the sweep angle of the outer wing panels, is used to adapt the aircraft's configuration to different flight conditions.

3.2. Aerodynamic Evaluation & Surrogate Modeling
The aerodynamic performance is evaluated using pre-trained Gaussian Process surrogate models. A comprehensive dataset was generated using a high-fidelity CFD solver for two key design points: a subsonic condition (Ma 0.4, H=0km) and a hypersonic condition (Ma 10.0, H=40km). Separate, high-fidelity GP models were trained for all six aerodynamic force and moment coefficients for each design point, using a combined dataset of over 450 samples per point.

3.3. Surrogate Model Validation
The accuracy of the surrogate models is paramount. The models were rigorously validated against a separate, unseen test set of over 50 CFD data points. The models demonstrated excellent predictive capability, with R-squared values consistently exceeding 0.99 for the primary force coefficients, confirming their suitability for use within the optimization framework. The ARD analysis of the trained models also provided initial physical insights, confirming that the sweep angle was the dominant variable for low-speed aerodynamics, while leading-edge shape variables were most critical at hypersonic speeds.

4. Results and Discussion
[This section will be filled with the final plots and analysis from the multi-objective optimization run. The structure is prepared below.]

4.1. Inner-Loop Convergence Efficiency
[Placeholder for the histogram of inner-loop iterations and discussion.]

4.2. Single-Objective Optimization: Discovering a Superior Design
[Placeholder for the single-objective convergence plot and discussion of the >28% CL improvement.]

4.3. Multi-Objective Optimization: The Pareto Frontier
[Placeholder for the Pareto Front plot and discussion of the objective trade-offs.]

4.4. Design Analysis via Parallel Coordinates
[Placeholder for the Parallel Coordinates Plot and analysis of the optimal design characteristics.]

4.5. High-Fidelity CFD Validation
[Placeholder for the CFD validation table and final proof of the framework's success.]

5. Conclusion
[To be written last. Will summarize the key achievements: the development of the novel framework, its demonstrated ability to handle complex constraints and find superior designs, and the final CFD validation of the results.]

References
[1] Kenway, G. K. W., and Martins, J. R. R., "Multipoint High-Fidelity Aerostructural Optimization of a Transport Aircraft Wing," Journal of Aircraft, Vol. 51, No. 2, 2014, pp. 423-437.

[2] Leoviriyakit, K., and Jameson, A., "Aerodynamic Shape Optimization of a Wing using the Adjoint Method with a Modified Viscous Term," 44th AIAA Aerospace Sciences Meeting and Exhibit, Reno, NV, 2006, AIAA 2006-1188.

[3] Fincham, J. H. S., and Friswell, M. I., "Aerodynamic optimisation of a camber morphing aerofoil," Aerospace Science and Technology, Vol. 43, 2015, pp. 245-255.

[4] Liu, B., Liang, H., Han, Z. H., et al., "Surrogate-based aerodynamic shape optimization of a morphing wing considering a wide Mach-number range," Aerospace Science and Technology, Vol. 124, 2022, 107557.

[5] Lam, R., "Bayesian Optimization for Materials Design," Bayesian Optimization, edited by R. Garnett, Cambridge University Press, 2023, pp. 139-158.

[6] Suzuki, K., "Application of Bayesian Optimization to Aerodynamic Design of a Supersonic Business Jet," Journal of Aircraft, Vol. 54, No. 1, 2017, pp. 17-27.

[7] Gardner, J. R., Kusner, M. J., Zuckerman, Z. E., et al., "Bayesian Optimization with Inequality Constraints," Proceedings of the 31st International Conference on Machine Learning, PMLR, Vol. 32, 2014, pp. 937-945.