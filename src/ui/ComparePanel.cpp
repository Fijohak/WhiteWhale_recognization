#include "ui/ComparePanel.h"

#include <algorithm>
#include <string>

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
        static_cast<float>(width);

    const float scaleY =
        maxHeight /
        static_cast<float>(height);


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
        ImTextureRef(textureId),
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
    images = newImages;
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
        std::move(callback);
}


void ComparePanel::draw()
{
    if (images.empty())
    {
        ImVec2 avail =
            ImGui::GetContentRegionAvail();


        const char* text =
            "No images in this group";


        const ImVec2 textSize =
            ImGui::CalcTextSize(text);


        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX()
            +
            std::max(
                0.0f,
                (
                    avail.x
                    -
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


        ImGui::TextUnformatted(text);

        return;
    }


    constexpr float minCardWidth =
        220.0f;

    constexpr float gap =
        12.0f;


    const float availableWidth =
        ImGui::GetContentRegionAvail().x;


    int columnCount =
        static_cast<int>(
            (
                availableWidth
                +
                gap
            )
            /
            (
                minCardWidth
                +
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
            gap
            *
            (
                columnCount - 1
            )
        )
        /
        columnCount;


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

    constexpr float imageHeight =
        190.0f;


    ImGui::PushID(index);


    ImGui::BeginChild(
        "##imageCard",
        ImVec2(
            cardWidth,
            cardHeight
        ),
        ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar
        |
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
                imageHeight
            );


        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX()
            +
            std::max(
                0.0f,
                (
                    innerWidth
                    -
                    drawSize.x
                )
                *
                0.5f
            )
        );


        ImGui::SetCursorPosY(
            ImGui::GetCursorPosY()
            +
            std::max(
                0.0f,
                (
                    imageHeight
                    -
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
            ImGui::IsItemHovered()
            &&
            ImGui::IsMouseClicked(
                ImGuiMouseButton_Left
            );


        if (ImGui::IsItemHovered())
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
    else
    {
        ImGui::InvisibleButton(
            "##emptyImage",
            ImVec2(
                innerWidth,
                imageHeight
            )
        );


        clicked =
            ImGui::IsItemClicked();


        const ImVec2 min =
            ImGui::GetItemRectMin();

        const ImVec2 max =
            ImGui::GetItemRectMax();


        ImGui::GetWindowDrawList()
            ->AddRect(
                min,
                max,
                ImGui::GetColorU32(
                    ImGuiCol_Border
                )
            );


        const char* text =
            "Image";


        const ImVec2 textSize =
            ImGui::CalcTextSize(text);


        ImGui::GetWindowDrawList()
            ->AddText(
                ImVec2(
                    min.x
                    +
                    (
                        max.x
                        -
                        min.x
                        -
                        textSize.x
                    )
                    *
                    0.5f,

                    min.y
                    +
                    (
                        max.y
                        -
                        min.y
                        -
                        textSize.y
                    )
                    *
                    0.5f
                ),

                ImGui::GetColorU32(
                    ImGuiCol_TextDisabled
                ),

                text
            );
    }


    ImGui::SetCursorPosY(
        imageHeight
        +
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


    if (
        clicked &&
        imageClick
    )
    {
        imageClick(index);
    }


    ImGui::EndChild();


    ImGui::PopID();
}
