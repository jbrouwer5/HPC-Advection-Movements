#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <mpi.h> 
using namespace std;

// helper functions for memory management
double* allocate_matrix(int rows, int cols){
    double* matrix = new double[rows*cols]; 
    return matrix;
}

void delete_matrix(double* matrix){
    delete[] matrix;
}

void initialize_matrix(double* matrix, int N, double L, int startrow, 
                        int endrow) 
{ 
    double d_y = L / (N-1); 
    double y0 = -1*L/2;
    int rows = endrow-startrow; 
    // initialize c_n with a rectangle of 1s 
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<N; j++)
        {
            matrix[(i+1)*(N+2)+(j+1)] = (abs(y0 + (startrow+i) * d_y) <= 0.1) ? 1.0 : 0.0;
        }
    }
}


// Needs slab updates
void print_matrix(int mype, int nprocs, int rows_per_slab, double* current, int N)
{
    // Print initial
    if (mype == 0){
        double* buffer_list[nprocs]; 
        for (int i=0; i<nprocs; i++)
        {
            buffer_list[i] = allocate_matrix(rows_per_slab+2, N+2);
        }

        for(int i = 0; i < (rows_per_slab+2)*(N+2); i++)
            buffer_list[0][i] = current[i]; // copy the allocated memory 

        ofstream myFile;
        myFile.open("output.txt", ofstream::app);

        myFile << "["; // outer brackets
        
        // for every square in this row 
        // receive that square_rows data
        for (int j=0; j<nprocs; j++)
        {
            if (j > 0){
                MPI_Status status;
                MPI_Recv(buffer_list[j], (N+2)*(rows_per_slab+2), MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &status); 
            }
        }

        // for every row of data in this squares
        for (int k=0; k<N; k++)
        {
            myFile << "[";
            // print the row from that slab
            for (int m=1; m<N+1; m++)
            {
                myFile << scientific << buffer_list[k/(rows_per_slab*(N+2))][(N+2)*k+m];  // print the value 
                // if (!(l==square_rows-1 && m == rows_per_square)){
                //     myFile << ",";
                // }
            }
            
            // if (k < rows_per_slab || i < rows_per_slab-1){
            //     myFile << "],";
            // }
            // else { myFile << "]"; }
        }
        

        myFile << "],\n";
        myFile.close();
        for (int i=1; i< nprocs; i++)
        {
            delete_matrix(buffer_list[i]); 
        }
    }
    else {
        MPI_Send(current, (N+2)*(rows_per_slab+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void lax_distributed(int N, double L, double T, int nprocs, int mype)
{
    int starty; int endy; 
    int u_neighbor; int d_neighbor; 
    int num_rows; int rows_per_slab; 

    // based on N and rank number, calculate bounds 
    rows_per_slab = ceil(double(N) / double(nprocs));
    starty = mype * rows_per_slab; 
    endy = (mype+1) * rows_per_slab; 

    if (endy > N) endy = N;

    num_rows = endy-starty; 

    u_neighbor = (mype+1) % nprocs; 
    d_neighbor = (mype-1 + nprocs) % nprocs; 

    // allocating grid for c_n and c_n+1
    double* current = allocate_matrix(num_rows+2,N+2); 
    double* next = allocate_matrix(num_rows+2,N+2); 

    // calculating delta x, delta t, and sigma 
    double d_x = L / (N-1); 
    double d_t = 1.25e-4; 

    // Given the current bounds, initialize the matrix
    initialize_matrix(current, N, L, starty, endy);
    // print_matrix(mype, nprocs, rows_per_slab, current, N); 

    double origin = -1*L/2;
    double* u = new double[N]; 
    double* v = new double[num_rows]; 
    double curr_u; double curr_v; 
    
    for (int i=1; i<num_rows+1; i++) 
    {
        curr_v = -1.0*sqrt(2.0)*(origin+(d_x*(starty+i)));
        v[i-1] = curr_v; 
    }

    for (int i=1; i<N+1; i++) 
    {
        curr_u = sqrt(2.0)*(origin+(d_x*(i+N))); 
        u[i-1] = curr_u; 
    }

    // send arrays for left and right boundaries 
    double* send_cols[2]; // 0=up, 1=down
    send_cols[0] = new double[N];  
    send_cols[1] = new double[N]; 

    // receive arrays // 0=up, 1=down
    double* ghost_cells[2];
    ghost_cells[0] = new double[N]; 
    ghost_cells[1] = new double[N]; 

    MPI_Status status;

    int full_cols = N+2; 
    double d_t_d_x_const = d_t / (2.0 * d_x); 
    // update matrix loop
    double start = omp_get_wtime(); 
    for (double k=0; k<T; k+=d_t) // iterate T with delta t step size
    {
        // if (k >= (T / 2) && (k-d_t) < (T / 2)) 
        //     print_matrix(mype, square_rows, rows_per_square, current);
 
        // fill send arrays with boundaries
        for (int i=1; i<num_rows+1; i++)
        {
            send_cols[1][i-1] = current[(num_rows)*(full_cols)+i]; // down row 
            send_cols[0][i-1] = current[(full_cols)+i]; // up row
        }

        for (int j=1; j<N+1; j++)
        {
            current[j] = current[num_rows*(full_cols)+j];
            current[(num_rows+1)*(full_cols)+j] = current[full_cols+j];
        }

        if (mype % 2 == 0)    
        {
            // receive up
            MPI_Recv(ghost_cells[0], N, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD, &status);
            // send up
            MPI_Send(send_cols[0], N, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD);
            // receive down
            MPI_Recv(ghost_cells[1], N, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD, &status);
            // send down
            MPI_Send(send_cols[1], N, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD);
        }
        else 
        {
            // send down
            MPI_Send(send_cols[1], N, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD);
            // receive down
            MPI_Recv(ghost_cells[1], N, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD, &status);
            // send up
            MPI_Send(send_cols[0], N, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD);
            // receive up
            MPI_Recv(ghost_cells[0], N, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD, &status);
        }

        // transfer up and down receives to current 
        for (int i=1; i<N+1; i++)
        {
            current[i] = ghost_cells[0][i-1]; // up
            current[(num_rows+1)*(full_cols)+i] = ghost_cells[1][i-1]; // down
        }

        #pragma omp parallel for default(none) shared(current,next,d_t_d_x_const,u,v,N,num_rows,full_cols) private(curr_v) schedule(dynamic)
        for (int i=1; i<num_rows+1; i++) // for every cell i,j
        {
            curr_v = v[i]; 
            for (int j=1; j<N+1; j++)
            {
                next[i*(N+2)+j] =  (current[(i+1)*(full_cols)+j] + 
                               current[(i-1)*(full_cols)+j] + 
                               current[i*(full_cols)+j-1] + 
                               current[i*(full_cols)+j+1]) / 4.0 
                                - d_t_d_x_const * 
                                (u[j] * (current[(i+1)*full_cols+j] - 
                                      current[(i-1)*full_cols+j]) +
                                 curr_v * (current[i*full_cols+j+1] - 
                                      current[i*full_cols+j-1]));
            }
        }
        
        double* tmp = current; 
        current = next; 
        next = tmp; 
    }
    double end = omp_get_wtime(); 
    if (mype==0){
       cout << "took " << end-start << " seconds\n"; 
    }
    // // MPI_Barrier(MPI_COMM_WORLD);
    // // print_matrix(mype, square_rows, rows_per_square, current); 
    // MPI_Barrier(MPI_COMM_WORLD);

    delete_matrix(current); delete_matrix(next); 
    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    // initialize MPI
    int nprocs; /* number of procs used in this invocation */
    int mype  ; /* my processor id (from 0 .. nprocs-1) */
    int stat;   /* used as error code for MPI calls */

    MPI_Init(&argc, &argv);  /* do this first to init MPI */
    /* when you run program you do as:                                             
        mpiexec -n <number_of_procs> <executable> <arg1> <arg2> ..                  
    /* after calling MPI_Init you may assume argv is correct */

    int N = 4000; 
    double L = 1.0; 
    double T = 1.0; 

    if (argc < 2)
    {
        cout << "Not enough arguments"; 
        return 0; 
    }

    int nt = stoi(argv[1]);
    omp_set_num_threads(nt); 

    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* return number of procs */
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); /* my integer proc id */

    lax_distributed(N, L, T, nprocs, mype); 

    return 0;
}