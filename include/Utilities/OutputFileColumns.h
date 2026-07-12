/*-----------------------------------------------------------------------*\
|    ___                   ____  __  __  ___  _  _______                  |
|   / _ \ _ __   ___ _ __ / ___||  \/  |/ _ \| |/ / ____| _     _         |
|  | | | | '_ \ / _ \ '_ \\___ \| |\/| | | | | ' /|  _| _| |_ _| |_       |
|  | |_| | |_) |  __/ | | |___) | |  | | |_| | . \| |__|_   _|_   _|      |
|   \___/| .__/ \___|_| |_|____/|_|  |_|\___/|_|\_\_____||_|   |_|        |
|        |_|                                                              |
|                                                                         |
|   Author: Alberto Cuoci <alberto.cuoci@polimi.it>                       |
|   CRECK Modeling Lab <https://www.creckmodeling.polimi.it>              |
|   Department of Chemistry, Materials, and Chemical Engineering          |
|   Politecnico di Milano                                                 |
|   P.zza Leonardo da Vinci 32, 20133 Milano                              |
|                                                                         |
|-------------------------------------------------------------------------|
|                                                                         |
|   This file is part of the OpenSMOKEpp library.                         |
|                                                                         |
|   Copyright (C) 2026 Alberto Cuoci.                                     |
|                                                                         |
|   OpenSMOKEpp is free software: you can redistribute it and/or modify   |
|   it under the terms of the GNU General Public License as published by  |
|   the Free Software Foundation, either version 3 of the License, or     |
|   (at your option) any later version.                                   |
|                                                                         |
|   OpenSMOKEpp is distributed in the hope that it will be useful,        |
|   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
|   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
|   GNU General Public License for more details.                          |
|                                                                         |
|   You should have received a copy of the GNU General Public License     |
|   along with OpenSMOKEpp. If not, see <https://www.gnu.org/licenses/>.  |
|                                                                         |
\*-----------------------------------------------------------------------*/

/**
 * @file OutputFileColumns.h
 * @brief Fixed-width ASCII table writer.
 */

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace MOM
{

/**
 * @class OutputFileColumns
 * @brief Fixed-width, column-aligned ASCII output file writer for tabular data.
 *
 * Columns are declared once, finalized with `Complete()`, then filled row by
 * row using `NewRow()` and chained `operator<<` insertions.  Header labels are
 * written as `"Label(N)"`, where `N` is the 1-based column index.
 *
 * @note This class is **not** thread-safe.  If multiple threads write
 *       diagnostics, each should own a separate instance.
 */
class OutputFileColumns
{
public:

    /**
     * @brief Default constructor.
     */
    OutputFileColumns();

    /**
     * @brief Constructs the writer and opens the output file.
     * @param file_name Path to the output file.
     */
    explicit OutputFileColumns(const std::filesystem::path& file_name);

    /**
     * @brief Constructs the writer and opens the output file.
     * @param file_name Path to the output file.
     */
    explicit OutputFileColumns(const std::string& file_name);

    /**
     * @brief Open the output file.
     * @param file_name Path to the output file.
     * @pre The writer must not already own an open stream. Call `Close()` before
     *      reusing the same object for another file.
     */
    void Open(const std::filesystem::path& file_name);

    /**
     * @brief Open the output file.
     * @param file_name Path to the output file.
     * @pre The writer must not already own an open stream. Call `Close()` before
     *      reusing the same object for another file.
     */
    void Open(const std::string& file_name);

    /**
     * @brief Close the output file and reset all internal state.
     */
    void Close();

    /**
     * @brief Declare one output column.
     *
     * @param label     Human-readable column label (e.g. `"Temperature(K)"`).
     * @param precision Number of significant digits for floating-point values
     *                  written to this column.
     * @pre Call after `Open()` and before `Complete()`.
     */
    void AddColumn(const std::string& label, unsigned int precision);

    /**
     * @brief Finalize column declarations and write the header row.
     *
     * @throws std::runtime_error If Complete() has already been called on this
     *                            instance, or if no columns have been declared.
     */
    void Complete();

    /**
     * @brief Terminate the current data row and advance to the next.
     *
     * @throws std::runtime_error If the number of values inserted since the
     *                            last row boundary does not match the number of
     *                            declared columns.
     */
    void NewRow();

    /**
     * @brief Insert one value into the current column slot and advance.
     *
     * @tparam T Arithmetic type.  Supported instantiations are explicitly
     *           provided in `OutputFileColumns.cpp`: `int`, `unsigned int`,
     *           `long`, `unsigned long`, `long long`, `unsigned long long`,
     *           `float`, `double`, and `long double`.
     *
     * @param  number The value to write.
     * @return Reference to `*this` for operator chaining.
     *
     * @pre The number of insertions in each row must match the number of
     *      declared columns.
     */
    template <typename T>
    OutputFileColumns& operator<<(const T& number);

private:

    /**
     * @brief Reset all members to their constructed defaults.
     */
    void Default();

private:

    std::ofstream fOut_; //!< Output file stream (scientific, left-aligned).

    unsigned int n_;        //!< Total number of declared columns (set by Complete()).
    unsigned int iterator_; //!< Index of the next column to be written [0, n_).

    unsigned int tab_;      //!< Minimum gap (characters) between a column tag and the next field start.
    unsigned int minimum_;  //!< Absolute minimum field width applied to every column regardless of tag or precision.

    std::vector<std::string>  tag_;       //!< Column tags as written in the header, formatted as "Label(N)".
    std::vector<unsigned int> width_;     //!< Computed field widths per column [characters].
    std::vector<unsigned int> precision_; //!< Per-column floating-point significant-digit count.

    bool                  status_completed_; //!< True after Complete() has been called; guards against re-initialisation.
    std::filesystem::path file_name_;        //!< Path of the currently open file (used in error messages).
};

} // namespace MOM
