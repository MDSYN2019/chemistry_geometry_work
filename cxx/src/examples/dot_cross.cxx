#include <iostream>
#include <Eigen/Dense>
 
using namespace Eigen;

int main() {
  Eigen::Vector3d v(1,2,3);
  Vector3d w(0,1,2);

    // I need to implement the bottom code to get the duagonalized matrix
  
  Matrix2f A;
  A << 1, 2, 2, 3;
  std::cout << "Here is the matrix A:\n" << A << std::endl;

  SelfAdjointEigenSolver<Matrix2f> eigensolver(A);
  if (eigensolver.info() != Success) abort();

  std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
  std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
	    << "corresponding to these eigenvalues:\n"
	    << eigensolver.eigenvectors() << std::endl;
  
  std::cout << "Dot product: " << v.dot(w) << std::endl;
  double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
  std::cout << "Dot product via a matrix product: " << dp << std::endl;
  std::cout << "Cross product:\n" << v.cross(w) << std::endl;

  // Computing eigenvalues and eigenvectors

  MatrixXd A = MatrixXd::Random(6,6);
  std::cout << "Here is a random 6x6 matrix, A:" << std::endl << A << std::endl << std::endl;
 
  EigenSolver<MatrixXd> es(A);
  std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;
  std::cout << "The matrix of eigenvectors, V, is:" << std::endl << es.eigenvectors() << endl << std::endl;
 
  complex<double> lambda = es.eigenvalues()[0];
  std::cout << "Consider the first eigenvalue, lambda = " << lambda << std::endl;
  VectorXcd v = es.eigenvectors().col(0);
  std::cout << "If v is the corresponding eigenvector, then lambda * v = " << std::endl << lambda * v << std::endl;
  std::cout << "... and A * v = " << std::endl << A.cast<complex<double> >() * v << std::endl << std::endl;
 
  MatrixXcd D = es.eigenvalues().asDiagonal();
  MatrixXcd V = es.eigenvectors();
  std::cout << "Finally, V * D * V^(-1) = " << std::endl << V * D * V.inverse() << std::endl;

}
