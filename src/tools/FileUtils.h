#pragma once

#include <filesystem>
#include <string>


class FileUtils
{
public:

    static bool copyFileUnique(
        const std::filesystem::path& source,
        const std::filesystem::path& targetFolder,
        std::filesystem::path& copiedPath,
        std::string& error
    );
};
