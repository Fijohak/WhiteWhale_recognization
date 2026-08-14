#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "tools/ImageLoader.h"


struct ImageInfo
{
    std::filesystem::path path;

    ImageTexture texture;
};


class ImageManager
{
public:

    ImageManager() = default;


    ~ImageManager();


    ImageManager(
        const ImageManager&
    ) = delete;


    ImageManager& operator=(
        const ImageManager&
    ) = delete;


    bool loadGroup(
        const std::filesystem::path& groupPath
    );


    void clear();


    int getImageCount() const;


    const ImageInfo* getImage(
        int index
    ) const;


    const std::vector<ImageInfo>&
    getImages() const;


    const std::string&
    getLastError() const;


private:

    std::vector<ImageInfo> images;

    std::string lastError;
};
