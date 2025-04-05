/*

------------------------
Last Updated: 28/06/2024
------------------------

Quantum Chemistry code inspired by the Crawford Group:

Link: https://github.com/CrawfordGroup/ProgrammingProjects/blob/master/Project%2301/project1-instructions.pdf

Using Eigensolver in eigen:

https://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html

*/

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <cmath>
#include <cassert>
#include <tuple> // To return multiple values from a c++ function
#include <unordered_map>
#include <variant> // what are we using variant for?
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// Importing Eigen libraries
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "geometry_functions.hpp"

void numerovMethod(double x_initial, double s, int m) {
  // Function to perform the Numerov method calculation
  double E;
  const double ss = s * s / 12;
  
  while (true) {
        std::cout << "Enter reduced energy Er (enter 1e10 to quit): ";
        std::cin >> E;
        if (E > 1e9) {
            std::cout << "Quitting" << std::endl;
            break;
        }

        int nn = 0;
        std::vector<double> x(m + 1), g(m + 1), p(m + 1);
        
        p[0] = 0;
        p[1] = 0.0001;
        x[0] = x_initial;
        x[1] = x_initial + s;
        g[0] = x[0] * x[0] - 2 * E;
        g[1] = x[1] * x[1] - 2 * E;

        for (int i = 1; i < m; i++) {
            x[i + 1] = x[i] + s;
            g[i + 1] = x[i + 1] * x[i + 1] - 2 * E;
            p[i + 1] = (-p[i - 1] + 2 * p[i] + 10 * g[i] * p[i] * ss + g[i - 1] * p[i - 1] * ss) / (1 - g[i + 1] * ss);

            if (p[i + 1] * p[i] < 0) {
                nn++;
            }
        }
        std::cout << "Er = " << E << " Nodes = " << nn << " Psir(xm) = " << p[m] << std::endl;
    }
}


/*
-- Molecular Geometry and Rotational Constant Analysis --
*/

int main(void) {

  spdlog::set_level(spdlog::level::debug); // Set logging level to debug
  auto main_logger = spdlog::stdout_color_mt("new_console");
  // Define a base directory path - where the main programming projects are 
  std::string base_directory = "ProgrammingProjects/";

  // Project 1 Paths
  std::string acetaldehyde = base_directory + "Project#01/input/acetaldehyde.dat";
  std::string allene = base_directory + "Project#01/input/allene.dat";
  std::string benzene = base_directory + "Project#01/input/benzene.dat";

  // Project 2 Paths
  std::string geometry_31cb = base_directory + "Project#02/input/3c1b_geom.txt";
  std::string geometry_benzene = base_directory + "Project#02/input/benzene_geom.txt";
  std::string geometry_H2O = base_directory + "Project#02/input/h2o_geom.txt";

  std::string hessian_31cb = base_directory + "Project#02/input/3c1b_hessian.txt";
  std::string hessian_benzene = base_directory + "Project#02/input/benzene_hessian.txt";
  std::string hessian_H2O = base_directory + "Project#02/input/h2o_hessian.txt";

  // Project 3 Paths
  std::string STO = base_directory + "Project#03/input/ch4/STO-3G";
  std::string h2oFF = base_directory + "Project#03/input/h2o";
   
  std::vector<std::vector<double> > molA; 
  std::vector<std::vector<double> > molB; 
  std::vector<std::vector<double> > molC; 

  // Project 2 
  std::vector<std::vector<double> > Project2a_31cb_Geometry;
  std::vector<std::vector<double> > Project2a_31cb_Hessian;
  std::vector<std::vector<double> > Project2a_benzene_Geometry;
  std::vector<std::vector<double> > Project2a_benzene_Hessian;
  double distance, angle, outofplaneangle, torsionalangle;
  
  // Project 1   
  // Read the coordinates of actaldehyde, allene and benzene
  read_coordinates(acetaldehyde, molA);
  read_coordinates(allene, molB);
  read_coordinates(benzene, molC);
    
  // Project 2 - reading in the files - Geometries and 
  read_coordinates(geometry_31cb, Project2a_31cb_Geometry);
  read_coordinates(geometry_benzene, Project2a_benzene_Geometry);
  read_coordinates(hessian_31cb, Project2a_31cb_Hessian);
  read_coordinates(hessian_benzene, Project2a_benzene_Hessian);  
  
  // Project 1 part - acetaldehyde part
  // ---- Compute the bond distances --- 
  distance = Rij(molA[0][1], molA[1][1], molA[0][2], molA[1][2], molA[0][3], molA[1][3]);  
  // ---- Compute the bond angles 
  angle = bond_angles(molA[0][1], molA[1][1], molA[2][1], molA[0][2], molA[1][2], molA[2][2], molA[0][3], molA[1][3], molA[2][3]);

  // ---- Compute the out of plane angle 
  outofplaneangle = out_of_plane_angle(molA[0][1], molA[1][1], molA[2][1], molA[3][1], molA[0][2], molA[1][2], molA[2][2], molA[3][2],  molA[0][3], molA[1][3], molA[2][3], molA[3][3]);
  std::tuple<double, double, double> COM;  
  COM = center_of_mass(molC);

  // --- Print out the torsional angle --- 
  torsionalangle = torsion_angle(molA[0][1], molA[1][1], molA[2][1], molA[3][1], molA[0][2], molA[1][2], molA[2][2], molA[3][2],  molA[0][3], molA[1][3], molA[2][3], molA[3][3]);

  /*
    ---- Project 2 ---- 
    
    This document outlines the fundamental theory and algorithmic steps for calculating the vibrational spectrum of a polyatomic molecule. 
    The essential theoretical step is the transfomration of the kinetic and potential energy operators from simple Cartesian displacement 
    coordinates to so called 'normal coordinates ' (linear combinations of mass-displacement coordinates), in which the Hamiltonian may be 
    written as a sum of independent harmonic oscillators      

  */
  
  // double iterator 
  std::vector<double>::iterator double_iterator;
  // hessian vector 
  std::vector<double> hessian;
  std::vector<std::vector<double> > initial_3c1b;
  std::vector<std::vector<double> > initial_3c1b_copy;
  std::vector<std::vector<double> > initial_hessian;
  std::string project_2_geometry_31cb = base_directory + "Project#02/input/3c1b_geom.txt";
  std::string project_2_hessian_31cb = base_directory + "Project#02/input/3c1b_hessian.txt";

  read_coordinates(project_2_geometry_31cb, initial_3c1b);
  read_coordinates(project_2_hessian_31cb, initial_hessian);
  std::vector<std::vector<double> > project2a_h2o_geometry;
  std::vector<std::vector<double> > project2a_h2o_hessian;
  std::unordered_map<int, int> h2o_masses;


  // Insert key-value pairs into the map
  h2o_masses[8] = 16;
  h2o_masses[1] = 1;
  std::string project_2_geometry_h2o = base_directory + "Project#02/input/h2o_geom.txt";
  std::string project_2_hessian_h2o = base_directory + "Project#02/input/h2o_hessian.txt";
  read_coordinates(project_2_geometry_h2o, project2a_h2o_geometry);
  read_coordinates(project_2_hessian_h2o, project2a_h2o_hessian);
  Eigen::MatrixXd h2o_hessian_eigen_matrix = reshape_vector(project2a_h2o_hessian, project2a_h2o_geometry.size(), project2a_h2o_geometry.size());
  Eigen::MatrixXd output = weigh_matrix(h2o_hessian_eigen_matrix, project2a_h2o_geometry, h2o_masses, project2a_h2o_geometry.size(), project2a_h2o_geometry.size());


  /*
    ---- Project 3 ----

    The purpose of this project is to provide a deeper understanding of Hartree Fock theory by demonstrating a simple implementation of the
    self-consistent field method. The theoretical background can be found in chapter 3 of the the text by Szabo and Ostlund. 

    
   */
  
  // Computing the Hartree Fock part - Reading the Nuclear Repulsion Energy, AO-basis overlap, kinetic-overlap, and
  // nuclear-attraction integrals.

  
  std::string project_3 = base_directory + "Project#03/input"; 
  std::vector<std::vector<double> > ao_basis_overlap;
  std::vector<std::vector<double> > kinetic_energy;
  std::vector<std::vector<double> > nuclear_attraction_integral;
  std::vector<std::vector<double> > two_electron_repulsion_integral;
  
  std::string s_dat = "ProgrammingProjects/Project#03/input/h2o/STO-3G/s.dat";
  std::string t_dat = "ProgrammingProjects/Project#03/input/h2o/STO-3G/t.dat";
  std::string v_dat = "ProgrammingProjects/Project#03/input/h2o/STO-3G/v.dat";
  std::string eri_dat = "ProgrammingProjects/Project#03/input/h2o/STO-3G/eri.dat";
  
  
  double nuclear_repulsion_data = 8.002367061810450; // initally applying the repulsion data 
  read_coordinates_hamiltonian(s_dat, ao_basis_overlap);
  read_coordinates_hamiltonian(t_dat, kinetic_energy);
  read_coordinates_hamiltonian(v_dat, nuclear_attraction_integral);
  read_coordinates_hamiltonian(eri_dat, two_electron_repulsion_integral);  
  Eigen::MatrixXd hamiltonian = compute_core_hamiltonian(kinetic_energy, nuclear_attraction_integral);

  std::cout << " " << hamiltonian << " " << std::endl;


  /*
    
    Orthogonalization of the Basis Set: The S^{-1/2} matrix
    --------------------------------------------------------

    Diagonalize the overlap matrix S

    SL_{s} = L_{s}V_{s}

    Where L_{s} is the matrix of eigenvectors and V_{s} is a diagonal
    matrix containing the corresponding eigenvalues.
    
  */

  
  Eigen::MatrixXd symmetric_orthogonal =  orthogonalize_basis_set(ao_basis_overlap);  
  std::cout << symmetric_orthogonal << std::endl;

  /*
    The Initial (Guess) Density Matrix
    ----------------------------------

    Form an initial Fock matrix, in the orthonormal basis using the core Hamiltonian:

    Diagonalize the Matrix

    Transform     
  */

  std::cout << std::endl;
  //std::cout << "eri.dat" << " " << two_electron_repulsion_integral << std::endl;
  Eigen::MatrixXd transposed_s = inital_density_matrix(symmetric_orthogonal, hamiltonian);
  std::cout << transposed_s << std::endl;
  
  return 0;

  
}
