#include "ui/ComparePanel.h"

#include <algorithm>
#include <utility>

#include "imgui.h"


namespace
{

ImVec2 fitImage(
    int width,
    int height,
    float maxWidth,
    float maxHeight
)
{
    if (
        width <= 0 ||
        height <= 0
    )
    {
        return ImVec2(
            maxWidth,
            maxHeight
        );
    }


    const float scaleX =
        maxWidth /
        static_cast<float>(
            width
        );


    const float scaleY =
        maxHeight /
        static_cast<float>(
            height
        );


    const float scale =
        std::min(
            scaleX,
            scaleY
        );


    return ImVec2(
        width * scale,
        height * scale
    );
}


void drawTexture(
    ImTextureID textureId,
    const ImVec2& size
)
{
#if IMGUI_VERSION_NUM >= 19200

    ImGui::Image(
        ImTextureRef(
            textureId
        ),
        size
    );

#else

    ImGui::Image(
        textureId,
        size
    );

#endif
}

}


void ComparePanel::setImages(
    const std::vector<UiImage>& newImages
)
{
    images =
        newImages;
}


void ComparePanel::clearImages()
{
    images.clear();
}


void ComparePanel::setImageClick(
    ImageClick callback
)
{
    imageClick =
        std::move(
            callback
        );
}


void ComparePanel::draw()
{
    // ==========================================
    // Empty
    // ==========================================

    if (images.empty())
    {
        const ImVec2 avail =
            ImGui::GetContentRegionAvail();


        const char* text =
            "No images in this group.";


        const ImVec2 textSize =
            ImGui::CalcTextSize(
                text
            );


        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX()
            +
            std::max(
                0.0f,
                (
                    avail.x -
                    textSize.x
                )
                *
                0.5f
            )
        );


        ImGui::SetCursorPosY(
            ImGui::GetCursorPosY()
            +
            30.0f
        );


        ImGui::TextUnformatted(
            text
        );


        return;
    }


    // ==========================================
    // Grid
    // ==========================================

    constexpr float minCardWidth =
        220.0f;


    constexpr float gap =
        12.0f;


    const float availableWidth =
        ImGui::GetContentRegionAvail().x;


    int columnCount =
        static_cast<int>(
            (
                availableWidth +
                gap
            )
            /
            (
                minCardWidth +
                gap
            )
        );


    columnCount =
        std::max(
            1,
            columnCount
        );


    const float cardWidth =
        (
            availableWidth
            -
            gap *
            static_cast<float>(
                columnCount - 1
            )
        )
        /
        static_cast<float>(
            columnCount
        );


    for (
        int i = 0;
        i < static_cast<int>(
            images.size()
        );
        ++i
    )
    {
        if (
            i > 0 &&
            i % columnCount != 0
        )
        {
            ImGui::SameLine(
                0.0f,
                gap
            );
        }


        drawImage(
            images[i],
            i,
            cardWidth
        );
    }
}


void ComparePanel::drawImage(
    const UiImage& image,
    int index,
    float cardWidth
)
{
    constexpr float cardHeight =
        240.0f;


    constexpr float imageAreaHeight =
        195.0f;


    ImGui::PushID(
        index
    );


    ImGui::BeginChild(
        "##imageCard",
        ImVec2(
            cardWidth,
            cardHeight
        ),
        ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse
    );


    const float innerWidth =
        ImGui::GetContentRegionAvail().x;


    bool clicked = false;


    if (image.valid())
    {
        const ImVec2 drawSize =
            fitImage(
                image.width,
                image.height,
                innerWidth,
                imageAreaHeight
            );


        // 水平居中
        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX()
            +
            std::max(
                0.0f,
                (
                    innerWidth -
                    drawSize.x
                )
                *
                0.5f
            )
        );


        // 垂直居中
        ImGui::SetCursorPosY(
            ImGui::GetCursorPosY()
            +
            std::max(
                0.0f,
                (
                    imageAreaHeight -
                    drawSize.y
                )
                *
                0.5f
            )
        );


        drawTexture(
            image.textureId,
            drawSize
        );


        clicked =
            ImGui::IsItemClicked(
                ImGuiMouseButton_Left
            );


        // Hover Border
        if (
            ImGui::IsItemHovered()
        )
        {
            ImGui::GetWindowDrawList()
                ->AddRect(
                    ImGui::GetItemRectMin(),
                    ImGui::GetItemRectMax(),
                    ImGui::GetColorU32(
                        ImGuiCol_ButtonHovered
                    )
                );
        }
    }


    // ==========================================
    // 文件名
    // ==========================================

    ImGui::SetCursorPosY(
        imageAreaHeight +
        12.0f
    );


    if (!image.name.empty())
    {
        ImGui::TextWrapped(
            "%s",
            image.name.c_str()
        );
    }
    else
    {
        ImGui::Text(
            "Image %d",
            index + 1
        );
    }


    // ==========================================
    // Click callback
    // ==========================================

    if (
        clicked &&
        imageClick
    )
    {
        imageClick(
            index
        );
    }


    ImGui::EndChild();


    ImGui::PopID();
}
