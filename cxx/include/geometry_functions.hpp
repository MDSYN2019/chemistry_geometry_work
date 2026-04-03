#ifndef GEOMETRY_FUNCTIONS_HPP
#define GEOMETRY_FUNCTIONS_HPP

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

void read_coordinates(const std::string& structure,
                      std::vector<std::vector<double>>& vec_input);

void read_coordinates_hamiltonian(
    const std::string& structure,
    std::vector<std::vector<double>>& vec_input);

void output_lines(std::vector<std::vector<double>>& vec_input);

std::tuple<double, double, double> center_of_mass(
    std::vector<std::vector<double>>& vec_input);

double Rij(const double& Xi,
           const double& Xj,
           const double& Yi,
           const double& Yj,
           const double& Zi,
           const double& Zj);

double bond_angles(const double& Xi,
                   const double& Xj,
                   const double& Xk,
                   const double& Yi,
                   const double& Yj,
                   const double& Yk,
                   const double& Zi,
                   const double& Zj,
                   const double& Zk);

double out_of_plane_angle(const double& Xi,
                          const double& Xj,
                          const double& Xk,
                          const double& Xl,
                          const double& Yi,
                          const double& Yj,
                          const double& Yk,
                          const double& Yl,
                          const double& Zi,
                          const double& Zj,
                          const double& Zk,
                          const double& Zl);

double torsion_angle(const double& Xi,
                     const double& Xj,
                     const double& Xk,
                     const double& Xl,
                     const double& Yi,
                     const double& Yj,
                     const double& Yk,
                     const double& Yl,
                     const double& Zi,
                     const double& Zj,
                     const double& Zk,
                     const double& Zl);

Eigen::Vector3d moments_of_inertia(std::vector<std::vector<double>>& vec_input);

Eigen::MatrixXd reshape_vector(std::vector<std::vector<double>>& input,
                               int rows,
                               int columns);

Eigen::MatrixXd weigh_matrix(Eigen::MatrixXd& hessian_eigen_matrix,
                             std::vector<std::vector<double>> geometry_input,
                             std::unordered_map<int, int>& atomic_masses,
                             int rows,
                             int columns);

Eigen::MatrixXd compute_core_hamiltonian(
    std::vector<std::vector<double>>& ke,
    std::vector<std::vector<double>>& nuclear_attraction_integral);

Eigen::MatrixXd orthogonalize_basis_set(
    std::vector<std::vector<double>>& overlap_matrix);

Eigen::MatrixXd inital_density_matrix(Eigen::MatrixXd& orthogonalized_basis_set_s,
                                      Eigen::MatrixXd& core_hamiltonian);

#endif
