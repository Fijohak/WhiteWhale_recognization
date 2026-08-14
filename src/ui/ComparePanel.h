#pragma once

#include <functional>
#include <vector>

#include "ui/UiImage.h"


class ComparePanel
{
public:

    using ImageClick =
        std::function<void(int)>;


    void draw();


    void setImages(
        const std::vector<UiImage>& images
    );


    void clearImages();


    void setImageClick(
        ImageClick callback
    );


private:

    void drawImage(
        const UiImage& image,
        int index,
        float cardWidth
    );


private:

    std::vector<UiImage> images;


    ImageClick imageClick;
};
