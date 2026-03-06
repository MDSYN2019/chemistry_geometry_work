#include <iostream>
#include <iomanip>
#include <spdlog/common.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <cmath>
#include <tuple>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// variant 
#include <variant>
// Eigen libraries
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_New;

class GeometryAnalysis {
public:
  int natom;
  int charge;
  int *zvals;
  double **geom;
  std::string point_group;
  
  void print_geom();
  void rotate(double phi);
  void translate(double x, double y, double z);
  double bond(int atom1, int atom2);
  double angle(int atom1, int atom2, int atom3);
  double torsion(int atom1, int atom2, int atom3, int atom4);

  GeometryAnalysis();
  ~GeometryAnalysis();
};


bool isSymmetric(const Eigen::MatrixXd& matrix, double epsilon = 1e-10) {
    if (matrix.rows() != matrix.cols()) {
        return false; // Only square matrices can be symmetric
    }
    return matrix.isApprox(matrix.transpose(), epsilon);
}


double Rij(double &Xi, double &Xj, double &Yi, double &Yj, double &Zi, double &Zj) {
  /*
    Compute interatomic distances here
  */
  double Rij = pow(pow(Xi - Xj, 2) + pow(Yi - Yj, 2) + pow(Zi - Zj, 2), 0.5);
  return Rij;
}

double bond_angles(double &Xi, double &Xj, double &Xk,  double &Yi, double &Yj, double &Yk,  double &Zi, double &Zj, double &Zk) {
  /*
    
   */
  double Rijval, Rjkval = 0.0;
  double eij_x, eij_y, eij_z;
  double ejk_x, ejk_y, ejk_z;
  
  Rijval = Rij(Xi, Xj, Yi, Yj, Zi, Zj);
  Rjkval = Rij(Xj, Xk, Yj, Yk, Zj, Zk); 
  eij_x = -(Xi - Xj)/Rijval;
  eij_y = -(Yi - Yj)/Rijval;
  eij_z = -(Zi - Zj)/Rijval;
  ejk_x = -(Xj - Xk)/Rjkval;
  ejk_y = -(Yj - Yk)/Rjkval;
  ejk_z = -(Zj - Zk)/Rjkval;
  
  Vector3d V1(eij_x, eij_y, eij_z);
  Vector3d V2(ejk_x, ejk_y, ejk_z); 
  double dot_product = V1.dot(V2); 
  // Inverse cos to remove the cos 
  dot_product = asin(dot_product);  
  return dot_product;
}

double out_of_plane_angle(double &Xi, double &Xj,
		       double &Xk, double &Xl,
		       double &Yi, double &Yj,
		       double &Yk, double &Yl,
		       double &Zi, double &Zj,
		       double &Zk, double &Zl) {

  double sin_denominator;
  double Rkjval, Rklval, Rkival = 0;  
  double ekj_x, ekj_y, ekj_z;
  double ekl_x, ekl_y, ekl_z;
  double eki_x, eki_y, eki_z;
  double ans = 0;

  sin_denominator = bond_angles(Xj, Xk, Xl, Yj, Yk, Yl, Zj, Zk, Zl);
  sin_denominator = sin(sin_denominator);
  
  Rkjval = Rij(Xk, Xj, Yk, Yj, Zk, Zj);
  Rklval = Rij(Xk, Xl, Yk, Yl, Zk, Zl);
  Rkival = Rij(Xk, Xi, Yk, Yi, Zk, Zi);


  // computing the unit vectors between atoms 
  ekj_x = -(Xk - Xj)/Rkjval;
  ekj_y = -(Yk - Yj)/Rkjval;
  ekj_z = -(Zk - Zj)/Rkjval;

  ekl_x = -(Xk - Xl)/Rklval;
  ekl_y = -(Yk - Yl)/Rklval;
  ekl_z = -(Zk - Zl)/Rklval;

  eki_x = -(Xk - Xi)/Rkival;
  eki_y = -(Yk - Yi)/Rkival;
  eki_z = -(Zk - Zi)/Rkival;

  Vector3d Vkj(ekj_x, ekj_y, ekj_z);
  Vector3d Vkl(ekl_x, ekl_y, ekl_z); 
  Vector3d Vki(eki_x, eki_y, eki_z); 
  Vector3d top = Vkj.cross(Vkl)/sin_denominator;
  ans = top.dot(Vki);
  ans = acos(ans);
  return ans;
}

double torsion_angle(double &Xi, double &Xj,
		     double &Xk, double &Xl,
		     double &Yi, double &Yj,
		     double &Yk, double &Yl,
		     double &Zi, double &Zj,
		     double &Zk, double &Zl) {
  /*
    Compute the torsion angle
  */
  double sindenominator_ijk, sindenominator_jkl;
  double Rkjval, Rklval, Rkival, Rijval = 0;  
  double eki_x, eki_y, eki_z; // unit vectors for k->i 
  double ekj_x, ekj_y, ekj_z; // unit vectors for k->j
  double ekl_x, ekl_y, ekl_z; // unit vectors for k->;
  double eij_x, eij_y, eij_z; // unit vectors for i->j;
  double ans = 0;
 
  sindenominator_ijk = bond_angles(Xi, Xj, Xk, Yi, Yj, Yk, Zi, Zj, Zk);
  sindenominator_jkl = bond_angles(Xj, Xk, Xl, Yj, Yk, Yl, Zj, Zk, Zl);
  sindenominator_ijk = sin(sindenominator_ijk);
  sindenominator_jkl = sin(sindenominator_jkl);

  Rkjval = Rij(Xk, Xj, Yk, Yj, Zk, Zj);
  Rklval = Rij(Xk, Xl, Yk, Yl, Zk, Zl);
  Rkival = Rij(Xk, Xi, Yk, Yi, Zk, Zi);
  Rijval = Rij(Xi, Xj, Yi, Yj, Zi, Zj);
  // compute the unit vectors - jk 
  ekj_x = -(Xk - Xj)/Rkjval;
  ekj_y = -(Yk - Yj)/Rkjval;
  ekj_z = -(Zk - Zj)/Rkjval;
  // compute the unit vectors - kl 
  ekl_x = -(Xk - Xl)/Rklval;
  ekl_y = -(Yk - Yl)/Rklval;
  ekl_z = -(Zk - Zl)/Rklval;
  // compute the unit vectors - ki
  eki_x = -(Xk - Xi)/Rkival;
  eki_y = -(Yk - Yi)/Rkival;
  eki_z = -(Zk - Zi)/Rkival;
  // compute the unit vectors - ij 
  eij_x = -(Xi - Xj)/Rijval;
  eij_y = -(Yi - Yj)/Rijval;
  eij_z = -(Zi - Zj)/Rijval;
  Vector3d vector_kj(ekj_x, ekj_y, ekj_z);
  Vector3d vector_kl(ekl_x, ekl_y, ekl_z); 
  Vector3d vector_ki(eki_x, eki_y, eki_z); 
  Vector3d vector_ij(eij_x, eij_y, eij_z); 
  Vector3d eij_cross_ejk = vector_ij.cross(vector_kj);
  Vector3d ejk_cross_ekl = vector_kj.cross(vector_kl);  
  ans = eij_cross_ejk.dot(ejk_cross_ekl);
  ans = ans / (sindenominator_ijk * sindenominator_jkl);
  ans = acos(ans);
  return ans;  
}

std::tuple<double, double, double> center_of_mass(std::vector<std::vector<double> >& VecInput) {
  /*
    Need explanation here 
  */
  double total_mass = 0; 
  double x_mass = 0;
  double y_mass = 0;
  double z_mass = 0;
  
  for (std::vector<std::vector<double> >::iterator it = VecInput.begin(); it != VecInput.end(); ++it) {    
    total_mass += (*it).at(0);
    x_mass += (*it).at(0) * (*it).at(1);
    y_mass += (*it).at(0) * (*it).at(2);
    z_mass += (*it).at(0) * (*it).at(3);
  }
  return std::make_tuple(x_mass/total_mass, y_mass/total_mass, z_mass/total_mass); 
}

void read_coordinates(std::string &structure, std::vector<std::vector<double> >& vec_input) {
  /*
    Read the coordinate/hessian files
  */  
  std::ifstream input(structure);
  std::vector<double> row;
  
  // If we don't have a filestream, fail with an error 
  if (!input.is_open()) {
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  std::getline(input, line);

  while (std::getline(input, line)) {
    std::istringstream is(line);
    std::vector<double> row((std::istream_iterator<double>(is)), std::istream_iterator<double>());
    vec_input.push_back(row);
  }
  
  input.close(); // always close file after reading
}

void read_coordinates_hamiltonian(std::string &structure, std::vector<std::vector<double> >& vec_input) {
  /*
    Read the coordinate/hessian files
  */  
  std::ifstream input(structure);
  std::vector<double> row;
  
  // If we don't have a filestream, fail with an error 
  if (!input.is_open()) {
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream is(line);
    std::vector<double> row((std::istream_iterator<double>(is)), std::istream_iterator<double>());
    vec_input.push_back(row);
  }
  
  input.close(); // always close file after reading
}

void moments_of_inertia(std::vector<std::vector<double> >& VecInput) {

  /*
    The moments of inertia are computed as follows, as listed here:

    Diagonal:
    --------
    
    I_{xx} = \sum_{i}{m_{i}(y_{i}^{2} + z_{i}^{2})} 
    I_{yy} = \sum_{i}{m_{i}(x_{i}^{2} + z_{i}^{2})} 
    I_{zz} = \sum_{i}{m_{i}(x_{i}^{2} + y_{i}^{2})} 

    Off-Diagonal:
    ------------
    I_{xy} = \sum_{i}{m_{i}(x_{i}y_{i}}
    I_{xz} = \sum_{i}{m_{i}(x_{i}z_{i}}
    I_{yz} = \sum_{i}{m_{i}(y_{i}z_{i}} 
        
  */
  
  double I_xx, I_yy, I_zz ;
  double I_xy, I_xz, I_yz;
  // Intertia Matrix 
  Eigen::MatrixXd I(3, 3);
  I.setZero(); 
  
  // Iterate through the input data

  for (size_t i = 0; i < VecInput.size(); ++i) {    
    double m_i = double(VecInput[i][0]);
    double x_i = VecInput[i][1];
    double y_i = VecInput[i][2];
    double z_i = VecInput[i][3];

    // Diagonal elements
    I_xx += m_i * (y_i * y_i + z_i * z_i);
    I_yy += m_i * (x_i * x_i + z_i * z_i);
    I_zz += m_i * (x_i * x_i + y_i * y_i);

    // Off-diagonal elements
    I_xy -= m_i * x_i * y_i;
    I_xz -= m_i * x_i * z_i;
    I_yz -= m_i * y_i * z_i;
  }
  //std::cout << "values for diagonal are: " << I_xx << " " << I_yy << " " << I_zz << std::endl;
  
  // Register diagonal elements
  I(0,0) = I_xx;
  I(1,1) = I_yy;
  I(2,2) = I_zz;
  // Register non-diagonal elements
  I(0,1) = I_xy;
  I(0,2) = I_xz;
  I(1,0) = I_xy;
  I(2,0) = I_xz;
  I(2,1) = I_yz;
  I(1,2) = I_yz;
  SelfAdjointEigenSolver<Matrix_New> solver(I);
  // after you have built the moment of inertia tensor, you may
  // compute its eigenvalues and eigenvalues as follows; 
  Matrix_New evecs = solver.eigenvectors();
  Matrix_New evals = solver.eigenvalues();
  MatrixXd diagonal_matrix = evals.asDiagonal();
  // Diagonalize the matrix: DiagonalMatrix = V * DiagonalMatrix * V^-1
  diagonal_matrix = evecs * diagonal_matrix * evecs.inverse();
  //return diagonalizedMatrix;
}



void output_lines(std::vector<std::vector<double> >& VecInput) {
  /*
    Takes a reference to a double vector of vectors and prints out the
    values within
  */
  spdlog::set_level(spdlog::level::debug); // Set logging level to debug
  auto console_logger = spdlog::stdout_color_mt("console");
  for (const auto& row: VecInput) {
    for (const auto& element : row) {
      console_logger->info("{}", element);
    }
  }
}

Eigen::MatrixXd reshape_vector(std::vector<std::vector<double> >& input, int rows, int columns) {
  /*
    The Hessian stored in memory should be a square matrix, while the format of the input file
    is rectangular. Understanding the translation between the two takes a bit of thinking
  */
  std::vector<double> flattened_hessian;
  for (const auto& innerVec: input) {
    for (const auto& element: innerVec) {
      flattened_hessian.push_back(element);
    }
  }

  Eigen::Map<Eigen::MatrixXd> hessian_eigen_matrix(flattened_hessian.data(), rows * 3, columns * 3);  
  return hessian_eigen_matrix;
}

Eigen::MatrixXd weigh_matrix(Eigen::MatrixXd& hessian_eigen_matrix,
		  std::vector<std::vector<double>> geometry_input, std::unordered_map<int, int>& atomic_masses, int rows, int columns) {
  /*
    create a separate mass matrix and multiply with the hessian eigen matrix
  */
  int initial_atomic_coordinate = 0;
  std::vector<double> flattened_weights;  
  for (size_t coordinates = 0; coordinates < 3; coordinates++) {  
    for (size_t i = 0; i < geometry_input.size() ; ++i) {
      for (size_t j = 0; j < geometry_input.size() ; ++j) {
	flattened_weights.push_back(std::sqrt(double(atomic_masses[geometry_input[initial_atomic_coordinate][0]]) * double(atomic_masses[geometry_input[j][0]])));
	flattened_weights.push_back(std::sqrt(double(atomic_masses[geometry_input[initial_atomic_coordinate][0]]) * double(atomic_masses[geometry_input[j][0]])));
	flattened_weights.push_back(std::sqrt(double(atomic_masses[geometry_input[initial_atomic_coordinate][0]]) * double(atomic_masses[geometry_input[j][0]])));
      }
    }
    initial_atomic_coordinate += 1;
  }
  
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weight_eigen_matrix(flattened_weights.data(), rows * 3, columns * 3);
  Eigen::MatrixXd result = hessian_eigen_matrix.array() / weight_eigen_matrix.array();
  return result;
}

Eigen::MatrixXd compute_core_hamiltonian(std::vector<std::vector<double> >& ke, std::vector<std::vector<double> >& nuclear_attraction_integral) {
  /*
    
   */
  int size = 0;
  std::vector<double> hamiltonian;
  for (size_t i = 0; i < ke.size() ; ++i) {
    hamiltonian.push_back(ke[i][2] + nuclear_attraction_integral[i][2]);
    // taking the max size and using it later to make a matrix from it 
    if (ke[i][0] >= size) {
      size = ke[i][0];
    }
  }
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weight_eigen_matrix(hamiltonian.data(), size, size);
  return weight_eigen_matrix;
}

Eigen::MatrixXd orthogonalize_basis_set(std::vector<std::vector<double> >& overlap_matrix) {
  /*
    Building the orthogonal matrix 
  */

  // insert the vector into the eigen matrix
  std::vector<double> s_vector;

  int S_size = 0; 

  for (int i = 0; i < overlap_matrix.size(); i++) {
    if (S_size <= overlap_matrix[i][0]) { 
      S_size = overlap_matrix[i][0];
    }
  }
  
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(S_size, S_size);
  std::cout << S_size << " " << std::endl;

  // fill in the matrix
  for (int i = 0; i < overlap_matrix.size(); i++) {
    //std::cout << overlap_matrix[i][0] << " " << overlap_matrix[i][1] << " " << overlap_matrix[i][2] << std::endl;
    S(int(overlap_matrix[i][0]-1), int(overlap_matrix[i][1]-1)) = overlap_matrix[i][2];
  }

  std::cout << "The diagonal matrix is " <<  S << std::endl;
  std::cout << std::endl;

  // Diagonalize the overlap matrix - building the solver for the eigen matrix 
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
  Eigen::MatrixXd LS = solver.eigenvectors(); // Matrix of eigenvectors
  Eigen::VectorXd eigenvalues = solver.eigenvalues(); // Vector of eigenvalues
  
  // Build the symmetric orthogonalization matrix S^-1/2
  Eigen::MatrixXd S_sqrt_inv = LS * eigenvalues.cwiseInverse().cwiseSqrt().asDiagonal() * LS.transpose();
  return S_sqrt_inv;
}


Eigen::MatrixXd inital_density_matrix(Eigen::MatrixXd &orthogonalized_basis_set_s, Eigen::MatrixXd &core_hamiltonian) {
  /*
    1. Form an Fock matrix, F_0 in the orthogonal basiss using the core hamiltonian
   */

  // first ensure that S is symmetric 
  auto symmetric = isSymmetric(orthogonalized_basis_set_s);
  if (symmetric) {
    std::cout << "The matrix is symmetric." << std::endl;
    } else {
    std::cout << "The matrix is not symmetric." << std::endl;
  }
 
  Eigen::MatrixXd transposed_S = orthogonalized_basis_set_s.transpose();
  // Step 2: Form the Fock matrix F_0 in the orthogonal basis using the core Hamiltonian
  Eigen::MatrixXd F_0 = core_hamiltonian;


  // Now we need to diagonalize F_0
  // Step 3: Diagonalize the Fock matrix to obtain eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(F_0);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvalue decomposition failed");
  }
  Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues();
  Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    
  return transposed_S;
}
