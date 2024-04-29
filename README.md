HPC Models to Solve Large-Scale Advection Equations 

Model Types
  - Serial  
        - To run Serial, I use ```clang++ -std=c++20 serial_advection.cpp``` on Mac.  
        - You can use your prefered c++ compiler. Then run your executable.  
  
  - Parallel Shared Memory  (OpenMP)  
        - To run Parallel Shared Memory, first download OpenMP on your system.  
        - Then, I use ```clang++ -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp multicore_advection.cpp```  
        where the include and library flags point to my OpenMP installation.  
        - Then run your executable with the number of threads you want, ```./a.out 16```  
        - ```g++ -fopenmp multicore_advection.cpp``` tends to work elsewhere.
    
  - Parallel Distributed Memory (MPI)  
      - Square Decomposition (Perfect Square # of Nodes only)   
          -  To run locally, download mpi.  
          - Then I compile with ```mpicxx shared_mem_advection.cpp``` on Mac and run with ```mpiexec -n 4 a.out```.  
          - The -n flag sets the number of nodes.  
            
      - Slab Decomposition  
          - To run locally, download mpi.  
          - Then I compile with ```mpicxx slab_distributed.cpp``` on Mac and run with ```mpiexec -n 4 a.out```.  
          - The -n flag sets the number of nodes.  
          
  - Parallel Distributed Memory Hybrid (OpenMP + MPI)  
      - Square Decomposition (Perfect Square # of Nodes only)   
          - To run locally, download OpenMP and MPI.  
          - I run ```mpicxx -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp hybrid_mpi.cpp```  
            where the include and library flags point to my OpenMP installation.  
          - I run my executable with ```mpiexec -n 4 a.out 16``` where this example would give 4 nodes and 16 threads.   
          
      - Slab Decomposition  
          - To run locally, download OpenMP and MPI.  
          - I run ```mpicxx -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp hybrid_slab.cpp``` where the include and library flags point to my OpenMP installation.  
          - I run my executable with ```mpiexec -n 4 a.out 16``` where this example would give 4 nodes and 16 threads.   

Empirical testing showed that the Square Decomposition OpenMP + MPI strategy is about 
92 times as fast as a single core implementation. 