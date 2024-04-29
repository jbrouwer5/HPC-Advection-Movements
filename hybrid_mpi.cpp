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
                        int endrow, int startcol, int endcol) 
{ 
    double d_y = L / (N-1); 
    double y0 = -1*L/2;
    int cols = endcol-startcol; 
    int rows = endrow-startrow; 
    // initialize c_n with a rectangle of 1s 
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            matrix[(i+1)*(cols+2)+(j+1)] = (abs(y0 + (startrow+i) * d_y) <= 0.1) ? 1.0 : 0.0;
        }
    }
}

void print_matrix(int mype, int square_rows, int rows_per_square, double* current)
{
    // Print initial
    if (mype == 0){
        double* buffer_list[square_rows]; 
        for (int i=0; i<square_rows; i++)
        {
            buffer_list[i] = allocate_matrix(rows_per_square+2, rows_per_square+2);
        }

        for(int i = 0; i < (rows_per_square+2)*(rows_per_square+2); i++)
            buffer_list[0][i] = current[i]; // copy the allocated memory 

        ofstream myFile;
        myFile.open("output.txt", ofstream::app);

        myFile << "["; // outer brackets
        // for each row of squares
        for (int i=0; i<square_rows;i++)
        {   
            // for every square in this row 
            // receive that square_rows data
            for (int j=0; j<square_rows; j++)
            {
                if (i > 0 || j > 0){
                    MPI_Status status;
                    MPI_Recv(buffer_list[j], (rows_per_square+2)*(rows_per_square+2), MPI_DOUBLE, i*square_rows+j, 0, MPI_COMM_WORLD, &status); 
                }
            }

            // for every row of data in this squares
            for (int k=1; k<rows_per_square+1; k++)
            {
                myFile << "[";
                // for every square that is a part of that row
                for (int l=0; l<square_rows; l++)
                {
                    // print the row from that square
                    for (int m=1; m<rows_per_square+1; m++)
                    {
                        myFile << scientific << buffer_list[l][(rows_per_square+2)*k+m];  // print the value 
                        if (!(l==square_rows-1 && m == rows_per_square)){
                            myFile << ",";
                        }
                    }
                }
                if (k < rows_per_square || i < square_rows-1){
                    myFile << "],";
                }
                else { myFile << "]"; }
            }
        }

        myFile << "],\n";
        myFile.close();
        for (int i=1; i< square_rows; i++)
        {
            delete_matrix(buffer_list[i]); 
        }
    }
    else {
        MPI_Send(current, (rows_per_square+2)*(rows_per_square+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void lax_distributed(int N, double L, double T, int nprocs, int mype)
{
    int startx; int endx; int starty; int endy; 
    int u_neighbor; int d_neighbor; int l_neighbor; int r_neighbor; 
    int rows_per_square; int square_rows; int num_rows; int num_cols;  

    // based on N and rank number, calculate bounds 
    if (sqrt(nprocs) == int(sqrt(nprocs))) {
        square_rows = int(sqrt(nprocs)); 
        rows_per_square = ceil(double(N) / double(square_rows));
        startx = (mype % square_rows) * rows_per_square; 
        endx = (mype % square_rows + 1) * rows_per_square; 
        starty = (mype / square_rows) * rows_per_square; 
        endy = (mype / square_rows + 1) * rows_per_square; 

        if (endx > N) endx = N; if (endy > N) endy = N;

        num_rows = endy-starty; 
        num_cols = endx-startx;  

        u_neighbor = (mype + square_rows) % nprocs;
        d_neighbor = (mype - square_rows + nprocs) % nprocs;
        l_neighbor = mype - 1 + ((mype % square_rows == 0) ? square_rows : 0); 
        r_neighbor = mype + 1 - ((mype % square_rows == square_rows-1) ? square_rows : 0);
    } 
    else {
        cout << "Not a perfect square";
        return;
    }

    // allocating grid for c_n and c_n+1
    double* current = allocate_matrix(num_rows+2,num_cols+2); 
    double* next = allocate_matrix(num_rows+2,num_cols+2);

    // calculating delta x, delta t, and sigma 
    double d_x = L / (N-1); 
    double d_t = 1.25e-4; 

    // Given the current bounds, initialize the matrix
    initialize_matrix(current, N, L, starty, endy, startx, endx);
    // print_matrix(mype, square_rows, rows_per_square, current); 

    double origin = -1*L/2;
    double* u = new double[num_cols]; 
    double* v = new double[num_rows]; 
    double curr_u; double curr_v; 
    
    for (int i=1; i<num_rows+1; i++) 
    {
        curr_v = -1.0*sqrt(2.0)*(origin+(d_x*(starty+i)));
        v[i-1] = curr_v; 
    }

    for (int i=1; i<num_cols+1; i++) 
    {
        curr_u = sqrt(2.0)*(origin+(d_x*(i+startx))); 
        u[i-1] = curr_u; 
    }

    // send arrays for left and right boundaries 
    double* send_cols[4]; // 0=up, 1=right, 2=down, 3=left
    send_cols[0] = new double[num_cols]; 
    send_cols[1] = new double[num_rows]; 
    send_cols[2] = new double[num_cols]; 
    send_cols[3] = new double[num_rows]; 

    // receive arrays // 0=up, 1=right, 2=down, 3=left
    double* ghost_cells[4];
    ghost_cells[0] = new double[num_cols]; 
    ghost_cells[1] = new double[num_rows]; 
    ghost_cells[2] = new double[num_cols]; 
    ghost_cells[3] = new double[num_rows];

    MPI_Status status;

    int full_cols = num_cols+2; 
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
            send_cols[3][i-1] = current[i*full_cols+1]; // left col to send left
            send_cols[1][i-1] = current[i*full_cols+num_cols]; // right col to send right
            send_cols[2][i-1] = current[full_cols+i]; // down row 
            send_cols[0][i-1] = current[(num_rows)*full_cols+i]; // up row
        }

        if (mype % 2 == 0)
        {
            // send left
            MPI_Send(send_cols[3], num_rows, MPI_DOUBLE, l_neighbor, 0, MPI_COMM_WORLD);
            // receive left
            MPI_Recv(ghost_cells[3], num_rows, MPI_DOUBLE, l_neighbor, 0, MPI_COMM_WORLD, &status);
            // receive right
            MPI_Recv(ghost_cells[1], num_rows, MPI_DOUBLE, r_neighbor, 0, MPI_COMM_WORLD, &status);
            // send right
            MPI_Send(send_cols[1], num_rows, MPI_DOUBLE, r_neighbor, 0, MPI_COMM_WORLD);
        }
        else
        {
            // receive right
            MPI_Recv(ghost_cells[1], num_rows, MPI_DOUBLE, r_neighbor, 0, MPI_COMM_WORLD, &status);
            // send right
            MPI_Send(send_cols[1], num_rows, MPI_DOUBLE, r_neighbor, 0, MPI_COMM_WORLD);
            // send left
            MPI_Send(send_cols[3], num_rows, MPI_DOUBLE, l_neighbor, 0, MPI_COMM_WORLD);
            // receive left
            MPI_Recv(ghost_cells[3], num_rows, MPI_DOUBLE, l_neighbor, 0, MPI_COMM_WORLD, &status);
        }

        if ((mype / square_rows) % 2 == 0)    
        {
            // receive up
            MPI_Recv(ghost_cells[0], num_cols, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD, &status);
            // send up
            MPI_Send(send_cols[0], num_cols, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD);
            // send down
            MPI_Send(send_cols[2], num_cols, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD);
            // receive down
            MPI_Recv(ghost_cells[2], num_cols, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD, &status);
        }
        else 
        {
            // send down
            MPI_Send(send_cols[2], num_cols, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD);
            // receive down
            MPI_Recv(ghost_cells[2], num_cols, MPI_DOUBLE, d_neighbor, 0, MPI_COMM_WORLD, &status);
            // receive up
            MPI_Recv(ghost_cells[0], num_cols, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD, &status);
            // send up
            MPI_Send(send_cols[0], num_cols, MPI_DOUBLE, u_neighbor, 0, MPI_COMM_WORLD);
        }

        // transfer left and right receives to current 
        for (int i=1; i<num_rows+1; i++)
        {
            current[i*full_cols] = ghost_cells[3][i-1]; // left col to send left
            current[i*full_cols+num_cols+1] = ghost_cells[1][i-1]; // right col to send right
        }

        // transfer up and down receives to current 
        for (int i=1; i<num_cols+1; i++)
        {
            current[i] = ghost_cells[2][i-1]; // down
            current[(num_rows+1)*full_cols+i] = ghost_cells[0][i-1]; // up
        }

        #pragma omp parallel for default(none) shared(current,next,d_t_d_x_const,u,v,N,num_rows,num_cols,full_cols) private(curr_v) schedule(dynamic)
        for (int i=1; i<num_rows+1; i++) // for every cell i,j
        {
            curr_v = v[i]; 
            for (int j=1; j<num_cols+1; j++)
            {
                next[i*full_cols+j] =  (current[(i+1)*full_cols+j] + 
                               current[(i-1)*full_cols+j] + 
                               current[i*full_cols+j-1] + 
                               current[i*full_cols+j+1]) / 4.0 
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
    
    // MPI_Barrier(MPI_COMM_WORLD);
    // print_matrix(mype, square_rows, rows_per_square, current); 
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