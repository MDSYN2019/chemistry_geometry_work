#include <varaint>
#include <string>
#include <cassert>
#include <vector>
#include <string>

/*

 
*/

std::vector<int> count_lines_in_files(const std::vector<std::string>& files) {

  std::vector<int> results;
  char c = 0;
  // for const file in files
  // of constant size and we want to autoamtically detect type 
  for (const auto& file: files) {
    int line_count  = 0;
    std::ifstream in(file);

    while (in.get(c)) {
      if (c == '\n') {
	line_count++;
      }
    } 
    results.push_back(line_count);
  }
  return results; 
}

// Using std::count to count newline characters

int count_lines(const std::string& filename) {

  std::ifstream in(filename);

  return std::count(std::istreambuf_iterator<char>(in),
		    std::istreambuf_iterator<char>(),
		    '\n');
}


/*
The benefit of writing in the functional style is that 
you define the intent instead of specifying how to do something - 
and is the aim of most techniques covered in this book.


 */
std::vector<int> count_lines_in_files(const std::vector<std::string>& files) {

}




int main() {

  std::variant<int, float> v, w;
  v = 12;
  int i = std::get<int>(v);
  w = std::get<int>(v);
  w = std::get<0>(v);
  w = v;

  
  
  return 0;
  
}
