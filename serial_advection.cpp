#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <map>
using namespace std;
 
// helper functions for memory management
double* allocate_matrix(int rows, int cols){
    double* matrix = new double[rows*cols]; 
    return matrix;
}

void delete_matrix(double* matrix){
    delete[] matrix;
}

// helper function to print the array to a file
void print_matrix(int size, double matrix[])
{
    ofstream myFile;
    myFile.open("output.txt", ofstream::app);

    myFile << "["; // outer brackets

    // iterate rows 
    for (int i=1; i<size-1; i++)
    {
        myFile << "["; // start of the row 
        for (int j=1; j<size-1; j++)
        {   
            myFile << scientific << matrix[i*size+j];  // print the value 
            if (j < size-2) myFile << ","; 
        }
        myFile << "]";
        if (i < size-2) myFile << ","; 
    }

    myFile << "],\n";
    myFile.close();
}

void initialize_matrix(double* matrix, double L, int N) 
{
    // N here is the original N plus 2

    double d_y = L / (N-1); 
    double y0 = -1*L/2;
    int cols = N; 
    int rows = N; 
    // initialize c_n with a rectangle of 1s 
    for (int i=1; i<N-1; i++)
    {
        for (int j=1; j<N-1; j++)
        {
            matrix[i*N+j] = (abs(y0 + (i-1) * d_y) <= 0.1) ? 1.0 : 0.0;
        }
    }
}

void lax_serial(int N, double L, double T)
{
    // allocating grid for c_n and c_n+1
    double* current = allocate_matrix(N+2,N+2); 
    double* next = allocate_matrix(N+2,N+2);

    // calculating delta x, delta t, and sigma 
    double d_x = L / (N-1); 
    double d_t = 1.25 * pow(10,-4);
    double origin = -1*L/2;
    initialize_matrix(current, L, N+2);
    // print_matrix(N+2, current);

    double* u = new double[N]; 
    double* v = new double[N]; 
    double curr_u; double curr_v; 
    
    for (int i=1; i<N+1; i++) // for every cell i,j
    {
        curr_u = sqrt(2.0)*(origin+(d_x*i)); 
        curr_v = -1.0*sqrt(2.0)*(origin+(d_x*i));
        v[i-1] = curr_v; 
        u[i-1] = curr_u; 
    }

    int full_row = N+2; 
    double d_t_d_x_const = d_t / (2.0 * d_x); 
    time_t start;
    time_t end;
    time(&start);
    for (double k=0; k<T; k+=d_t) // iterate T with delta t step size
    {
        // if (k >= (T / 2) && (k-d_t) < (T / 2)){
        //     print_matrix(N+2, current);
        // }

        for (int i=1; i<N+1; i++)
        {
            current[i*(full_row)] = current[i*(full_row)+N];
            current[i*(full_row)+N+1] = current[i*(full_row)+1];
        }

        for (int j=1; j<N+1; j++)
        {
            current[j] = current[N*(full_row)+j];
            current[(N+1)*(full_row)+j] = current[full_row+j];
        }
         
        for (int i=1; i<N+1; i++) // for every cell i,j
        {
            curr_v = v[i]; 

            for (int j=1; j<N+1; j++)
            {
                next[i*(full_row)+j] =  ((current[(i+1)*(full_row)+j] + current[(i-1)*(full_row)+j] + 
                                     current[i*(full_row)+j-1] + current[i*(full_row)+j+1]) / 4.0)
                                - (d_t_d_x_const * 
                                (u[j] * (current[(i+1)*(full_row)+j] - 
                                      current[(i-1)*(full_row)+j]) +
                                 curr_v * (current[i*(full_row)+j+1] - 
                                      current[i*(full_row)+j-1])));
            }
        }
        
        double* tmp = current; 
        current = next; 
        next = tmp; 
    }
    time(&end);
    cout << "elapsed time: " << difftime(end, start) << "s" << endl;

    // print_matrix(N+2, current);
    delete_matrix(current); delete_matrix(next); 
}

int main(int argc, char* argv[])
{
    int N = 4000; 
    double L = 1.0; 
    double T = 1.0; 
    lax_serial(N, L, T); 
    
    return 0;
}