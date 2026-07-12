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

#include "Utilities/OutputFileColumns.h"

#include <stdexcept>

/**
 * @file OutputFileColumns.cpp
 * @brief Implementation of the fixed-width ASCII table writer.
 */

namespace MOM
{

OutputFileColumns::OutputFileColumns()
{
    Default();
}

OutputFileColumns::OutputFileColumns(const std::filesystem::path& file_name)
{
    Default();
    Open(file_name);
    file_name_ = file_name;
}

OutputFileColumns::OutputFileColumns(const std::string& file_name)
{
    Default();
    Open(file_name);
    file_name_ = file_name;
}

void OutputFileColumns::Default()
{
    n_                = 0;
    iterator_         = 0;
    minimum_          = 16;
    tab_              = 4;
    status_completed_ = false;
    file_name_.clear();

    tag_.clear();
    width_.clear();
    precision_.clear();
}

void OutputFileColumns::Open(const std::filesystem::path& file_name)
{
    file_name_ = file_name;

    fOut_.open(file_name, std::ios::out);
    fOut_.setf(std::ios::scientific);
    fOut_.setf(std::ios::left);
}

void OutputFileColumns::Open(const std::string& file_name)
{
    file_name_ = file_name;

    fOut_.open(file_name.c_str(), std::ios::out);
    fOut_.setf(std::ios::scientific);
    fOut_.setf(std::ios::left);
}

void OutputFileColumns::Close()
{
    if (fOut_.is_open())
        fOut_.close();

    Default();
}

void OutputFileColumns::AddColumn(const std::string& label, const unsigned int precision)
{
    iterator_++;
    std::stringstream number;
    number << iterator_;
    const std::string tag = label + "(" + number.str() + ")";
    tag_.push_back(tag);

    // Field width accounts for the header tag and a scientific-notation token.
    const unsigned int label_width     = static_cast<unsigned int>(tag.size()) + tab_;
    const unsigned int precision_width = precision + 9u;
    width_.push_back(std::max(label_width, std::max(precision_width, minimum_)));
    precision_.push_back(precision);
}

void OutputFileColumns::Complete()
{
    if (status_completed_)
    {
        std::stringstream message;
        message << "Filename: " << file_name_ << '\n'
                << "OutputFileColumns: Complete() has already been called; "
                   "columns cannot be re-declared.";
        throw std::runtime_error(message.str());
    }

    if (iterator_ == 0)
    {
        std::stringstream message;
        message << "Filename: " << file_name_ << '\n'
                << "OutputFileColumns: no columns have been declared via AddColumn().";
        throw std::runtime_error(message.str());
    }

    status_completed_ = true;
    n_                = iterator_;
    iterator_         = 0;

    for (unsigned int i = 0; i < n_; i++)
        fOut_ << std::setprecision(precision_[i]) << std::setw(width_[i]) << tag_[i];

    fOut_ << std::endl;
}

void OutputFileColumns::NewRow()
{
    // The first call may be used to start the first row.
    if (iterator_ != n_ && iterator_ != 0)
    {
        std::stringstream message;
        message << "Filename: " << file_name_ << '\n'
                << "OutputFileColumns: row closed with " << iterator_
                << " value(s) but " << n_ << " column(s) were declared.";
        throw std::runtime_error(message.str());
    }
    fOut_ << std::endl;
    iterator_ = 0;
}

template <typename T>
OutputFileColumns& OutputFileColumns::operator<<(const T& number)
{
    fOut_ << std::setprecision(precision_[iterator_]) << std::setw(width_[iterator_]) << number;
    iterator_++;
    return *this;
}

// Explicit instantiations for supported numeric types.
template OutputFileColumns& OutputFileColumns::operator<<(const int&);
template OutputFileColumns& OutputFileColumns::operator<<(const unsigned int&);
template OutputFileColumns& OutputFileColumns::operator<<(const long&);
template OutputFileColumns& OutputFileColumns::operator<<(const unsigned long&);
template OutputFileColumns& OutputFileColumns::operator<<(const long long&);
template OutputFileColumns& OutputFileColumns::operator<<(const unsigned long long&);

template OutputFileColumns& OutputFileColumns::operator<<(const float&);
template OutputFileColumns& OutputFileColumns::operator<<(const double&);
template OutputFileColumns& OutputFileColumns::operator<<(const long double&);

} // namespace MOM
