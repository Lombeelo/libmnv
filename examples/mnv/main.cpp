#define MNV_ERRORS_INCLUDE_MESSAGES
#include <mnv/mnv.hpp>

#include <iostream>
#include <vector>

template <typename Matr>
void printMatrix(Matr matr)
{
    for (size_t i = 0; i < matr.size(); i++)
    {
        std::cout << "{ ";
        for (auto &&item : matr[i])
        {
            std::cout << item << ", ";
        }
        std::cout << "}" << std::endl;
    }
}

int main(int, char *[])
{
    // sample data
    std::vector<mnv::valueVector<float, 5>> samples{
        {{5, 3.5f, 20, 3.6f, 22},
         {7, 3.11f, 25, 3.101f, 27},
         {9, 4.1f, 26, 4.3f, 28},
         {11, 4.7f, 32, 4.7f, 32},
         {13, 4.11f, 35, 4.11f, 40},
         {15, 5.1f, 40, 5.2f, 45},
         {17, 5.2f, 45, 5.4f, 50},
         {19, 5.3f, 48, 5.7f, 55},
         {21, 5.5f, 50, 5.9f, 64},
         {23, 5.55f, 51, 5.9f, 67},
         {25, 5.55f, 55, 5.9f, 70}}};
    auto covariance = mnv::calculateCovarianceMatrix(samples);
    auto mean = mnv::calculateMeanVector(samples);

    auto gen_packed = mnv::MNVGenerator<float, 5>::build(covariance, mean, 0);
    // Error handling
    if (std::holds_alternative<mnv::MNVGeneratorBuildError>(gen_packed))
    {
        auto err = std::get<mnv::MNVGeneratorBuildError>(gen_packed);
        std::cerr << err.message;
        return 1;
    }

    auto generator = std::get<mnv::MNVGenerator<float, 5>>(gen_packed);
    std::cout << "Generating samples:" << std::endl;
    std::vector<mnv::valueVector<float, 5>> values;
    for (size_t i = 0; i < 100000; i++)
    {
        values.push_back(generator.nextValue());
    }

    // printMatrix(values);

    std::cout << "covariance before: " << std::endl;

    printMatrix(covariance);
    std::cout << "covariance after: " << std::endl;
    printMatrix(mnv::calculateCovarianceMatrix(values));

    return 0;
}
