#ifndef MNV_HPP
#define MNV_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string_view>
#include <variant>
#include <vector>

#ifdef MNV_ERRORS_INCLUDE_MESSAGES
#define ERRMSG(str) str
#else
#define ERRMSG(str)
#endif

/**
 * @file mnv.hpp Main interface file of the mnv library
 * @author Dmitry Parfenyuk (cekunda.rf@gmail.com)
 * @brief A header-only library to generate random multivariate normal values
 * @version 1.0.0
 * @date 2023-07-08
 *
 * @copyright Copyright (c) 2023 Dmitry Parfenyuk
 *
 */

/**
 * @brief The mnv namespace
 *
 */
namespace mnv
{
    /**
     * @brief Basic vector, can be abstracted away
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Length of array, can be abstracted
     */
    template <typename T, size_t Dim>
    using valueVector = std::array<T, Dim>;

    /**
     * @brief Square matrix, array of arrays, size is statically defined
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Matrix size
     */
    template <typename T, size_t Dim>
    using MatrixSq = valueVector<valueVector<T, Dim>, Dim>;

    /**
     * @brief Struct to signal, that the generator build process was failed
     *
     */
    struct MNVGeneratorBuildError
    {
        /**
         * @brief Error type. Can be matched against to make error handling more straightforward
         *
         */
        enum class type
        {
            CovarianceMatrixIsNotPositiveDefinite,
            CovarianceMatrixIsNotSymmetric,
        };
        /**
         * @brief Field that holds the error type
         *
         */
        type type;
#ifdef MNV_ERRORS_INCLUDE_MESSAGES
        /**
         * @brief Error message. You can use them to print the error
         *
         */
        std::string_view message;
#endif
    };

    /**
     * @brief The main Generator class. It incapsulates the internal rng state and distribution parameters
     *
     *
     * @tparam T Type of values generated
     * @tparam Dim Dimension count of values
     */
    template <typename T, size_t Dim>
    class MNVGenerator
    {
    public:
        /**
         * @brief Generate the next value of rng.
         *
         * @return valueVector<T, Dim> Generated value
         */
        valueVector<T, Dim> nextValue();

        /**
         * @brief Main build fuction
         *
         * @param covariance Covariance matrix. MUST be positive-definite and symmetric.
         * @param mean Mean vector.
         * @param seed Starting internal rng seed.
         * @return std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError> \n
         *          If error happened, variant will contain MNVGeneratorBuildError. \n
         *          Else, there will be an instance of MNVGenerator. \n
         *          To properly check for errors, you should always check with std::holds_alternative<mnv::MNVGeneratorBuildError>() \n
         *          before std::get<>()'ing the generator \n
         *          See also <a href="https://en.cppreference.com/w/cpp/utility/variant">std::variant [cppreference.com]</a> \n
         *          You can also check tests and examples for usage.
         */
        static std::variant<MNVGenerator<T, Dim>, MNVGeneratorBuildError>
        build(
            MatrixSq<T, Dim> const &covariance,
            valueVector<T, Dim> const &mean,
            size_t seed);

    private:
        // private constructor is used to force MNVGenerator::build()
        MNVGenerator(MatrixSq<T, Dim> decomposedCovariance, valueVector<T, Dim> mean, size_t seed);

        // distribution params
        MatrixSq<T, Dim> m_decomposedCovariance{};
        valueVector<T, Dim> m_mean{};

        // rng params
        size_t m_seed{0};
        std::mt19937 m_generator{};
        std::normal_distribution<T> distribution{0, 1};
    };

    /**
     * @brief Function to statistically calculate covariance matrix using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Matrix size
     * @param input_vectors Input vectors to calculate covariance matrix
     * @return MatrixSq<T, Dim> The covariance matrix
     */
    template <typename T, size_t Dim>
    MatrixSq<T, Dim> calculateCovarianceMatrix(std::vector<valueVector<T, Dim>> const &input_vectors);

    /**
     * @brief Function to statistically calculate mean vector using statistic data
     *
     * @tparam T Underlying type, supposedly float/decimal
     * @tparam Dim Vectors' size
     * @param input_vectors Input vectors to calculate mean vector
     * @return valueVector<T, Dim> The mean vector
     */
    template <typename T, size_t Dim>
    valueVector<T, Dim> calculateMeanVector(std::vector<valueVector<T, Dim>> const &input_vectors);

} // namespace mnv

#include <mnv/mnv-impl.hpp>

#endif // MNV_HPP