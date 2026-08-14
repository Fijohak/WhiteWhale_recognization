#include "tools/FileUtils.h"

#include <system_error>


namespace fs = std::filesystem;


bool FileUtils::copyFileUnique(
    const fs::path& source,
    const fs::path& targetFolder,
    fs::path& copiedPath,
    std::string& error
)
{
    error.clear();

    copiedPath.clear();


    std::error_code ec;


    // ==========================================
    // Source
    // ==========================================

    if (
        !fs::exists(
            source,
            ec
        )
    )
    {
        error =
            "Source file does not exist.";

        return false;
    }


    if (
        !fs::is_regular_file(
            source,
            ec
        )
    )
    {
        error =
            "Source path is not a file.";

        return false;
    }


    // ==========================================
    // Target Folder
    // ==========================================

    if (
        !fs::exists(
            targetFolder,
            ec
        )
    )
    {
        error =
            "Target folder does not exist.";

        return false;
    }


    if (
        !fs::is_directory(
            targetFolder,
            ec
        )
    )
    {
        error =
            "Target path is not a folder.";

        return false;
    }


    // ==========================================
    // Target Filename
    // ==========================================

    fs::path targetPath =
        targetFolder /
        source.filename();


    int suffix = 1;


    // 已经存在同名文件时，
    // 不覆盖。
    while (
        fs::exists(
            targetPath,
            ec
        )
    )
    {
        fs::path newName =
            std::to_string(
                suffix
            )
            + "_";


        // operator += 不会添加路径分隔符，
        // 正好用于：
        //
        // 1_ + image.jpg
        newName +=
            source.filename();


        targetPath =
            targetFolder /
            newName;


        ++suffix;
    }


    // ==========================================
    // Copy
    // ==========================================

    ec.clear();


    if (
        !fs::copy_file(
            source,
            targetPath,
            fs::copy_options::none,
            ec
        )
    )
    {
        error =
            ec
                ? ec.message()
                : "Failed to copy file.";

        return false;
    }


    copiedPath =
        targetPath;


    return true;
}
