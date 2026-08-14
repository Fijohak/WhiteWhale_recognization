#include "app/ImageManager.h"

#include <algorithm>
#include <system_error>
#include <utility>


namespace fs = std::filesystem;


ImageManager::~ImageManager()
{
    clear();
}


bool ImageManager::loadGroup(
    const fs::path& groupPath
)
{
    lastError.clear();


    std::error_code error;


    // ==========================================
    // 检查 Group 目录
    // ==========================================

    if (
        !fs::exists(
            groupPath,
            error
        )
    )
    {
        lastError =
            "Group folder does not exist.";

        return false;
    }


    if (
        !fs::is_directory(
            groupPath,
            error
        )
    )
    {
        lastError =
            "Group path is not a folder.";

        return false;
    }


    // ==========================================
    // 先找到所有图片文件
    // ==========================================

    std::vector<fs::path>
        imagePaths;


    try
    {
        for (
            const auto& entry :
            fs::directory_iterator(
                groupPath,
                fs::directory_options::
                    skip_permission_denied
            )
        )
        {
            if (
                !entry.is_regular_file()
            )
            {
                continue;
            }


            if (
                !ImageLoader::isImageFile(
                    entry.path()
                )
            )
            {
                continue;
            }


            imagePaths.push_back(
                entry.path()
            );
        }
    }
    catch (
        const fs::filesystem_error& exception
    )
    {
        lastError =
            exception.what();

        return false;
    }


    // ==========================================
    // 稳定排序
    // ==========================================

    std::sort(
        imagePaths.begin(),
        imagePaths.end()
    );


    // ==========================================
    // 加载新图片之前释放旧 Group
    // ==========================================

    clear();


    images.reserve(
        imagePaths.size()
    );


    // ==========================================
    // 加载所有 Texture
    // ==========================================

    for (
        const auto& imagePath :
        imagePaths
    )
    {
        ImageInfo image;


        image.path =
            imagePath;


        std::string errorMessage;


        if (
            !ImageLoader::load(
                imagePath,
                image.texture,
                errorMessage
            )
        )
        {
            // 某一张图片损坏时，
            // 不影响整个 Group。
            //
            // 直接跳过。
            continue;
        }


        images.push_back(
            std::move(image)
        );
    }


    return true;
}


void ImageManager::clear()
{
    for (auto& image : images)
    {
        ImageLoader::release(
            image.texture
        );
    }


    images.clear();
}


int ImageManager::getImageCount() const
{
    return static_cast<int>(
        images.size()
    );
}


const ImageInfo*
ImageManager::getImage(
    int index
) const
{
    if (
        index < 0 ||
        index >= getImageCount()
    )
    {
        return nullptr;
    }


    return &images[index];
}


const std::vector<ImageInfo>&
ImageManager::getImages() const
{
    return images;
}


const std::string&
ImageManager::getLastError() const
{
    return lastError;
}
