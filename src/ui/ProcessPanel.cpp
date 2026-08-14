#include "ui/ProcessPanel.h"

#include <algorithm>

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


void ProcessPanel::setMode(
    ProcessMode newMode
)
{
    mode = newMode;
}


ProcessMode ProcessPanel::getMode() const
{
    return mode;
}


void ProcessPanel::resetMode()
{
    mode =
        ProcessMode::Select;
}


void ProcessPanel::setSinglePreview(
    const UiImage& image
)
{
    singlePreview = image;
}


void ProcessPanel::clearSinglePreview()
{
    singlePreview = {};
}


void ProcessPanel::setBatchPreview(
    const UiImage& image
)
{
    batchPreview = image;
}


void ProcessPanel::clearBatchPreview()
{
    batchPreview = {};
}


void ProcessPanel::handleFileDrop(
    const std::string& path
)
{
    if (
        mode == ProcessMode::Single
        &&
        singleDrop
    )
    {
        singleDrop(path);
    }
}


void ProcessPanel::setModeChange(
    ModeChange callback
)
{
    modeChange =
        std::move(callback);
}


void ProcessPanel::setPickSingle(
    Action callback
)
{
    pickSingle =
        std::move(callback);
}


void ProcessPanel::setSingleDrop(
    FileDrop callback
)
{
    singleDrop =
        std::move(callback);
}


void ProcessPanel::setSingleConfirm(
    Action callback
)
{
    singleConfirm =
        std::move(callback);
}


void ProcessPanel::setPickFolder(
    Action callback
)
{
    pickFolder =
        std::move(callback);
}


void ProcessPanel::setBatchPrev(
    Action callback
)
{
    batchPrev =
        std::move(callback);
}


void ProcessPanel::setBatchConfirm(
    Action callback
)
{
    batchConfirm =
        std::move(callback);
}

void ProcessPanel::setNewCategory(
    Action callback
)
{
    newCategory =
        std::move(callback);
}


void ProcessPanel::setBatchNext(
    Action callback
)
{
    batchNext =
        std::move(callback);
}


void ProcessPanel::draw()
{
    // ==========================================
    // Mode Select
    // ==========================================

    if (mode == ProcessMode::Select)
    {
        drawModeSelect();

        return;
    }


    // ==========================================
    // Back
    // ==========================================

    drawBackButton();


    // 如果刚刚点击 Back，
    // mode 已经变成 Select
    if (mode == ProcessMode::Select)
    {
        return;
    }


    ImGui::Separator();


    // ==========================================
    // Content
    // ==========================================

    constexpr float bottomHeight =
        56.0f;


    const float availableHeight =
        ImGui::GetContentRegionAvail().y;


    const float bodyHeight =
        std::max(
            0.0f,
            availableHeight
            -
            bottomHeight
        );


    ImGui::BeginChild(
        "##processBody",
        ImVec2(
            0.0f,
            bodyHeight
        ),
        ImGuiChildFlags_Borders
    );


    if (mode == ProcessMode::Single)
    {
        drawSingle();
    }
    else
    {
        drawBatch();
    }


    ImGui::EndChild();


    // ==========================================
    // Bottom Buttons
    // ==========================================

    if (mode == ProcessMode::Single)
    {
        drawSingleButtons();
    }
    else
    {
        drawBatchButtons();
    }
}

void ProcessPanel::drawBackButton()
{
    constexpr float buttonWidth =
        90.0f;

    constexpr float buttonHeight =
        32.0f;


    if (
        ImGui::Button(
            "< Back",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        mode =
            ProcessMode::Select;


        if (modeChange)
        {
            modeChange(mode);
        }
    }
}

void ProcessPanel::drawModeSelect()
{
    const ImVec2 avail =
        ImGui::GetContentRegionAvail();


    constexpr float buttonWidth =
        132.0f;

    constexpr float buttonHeight =
        44.0f;

    constexpr float gap =
        18.0f;


    const float rowWidth =
        buttonWidth
        *
        2.0f
        +
        gap;


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                avail.x
                -
                rowWidth
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
                avail.y
                -
                buttonHeight
            )
            *
            0.5f
        )
    );


    if (
        ImGui::Button(
            "Single Image",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        mode =
            ProcessMode::Single;


        if (modeChange)
        {
            modeChange(mode);
        }
    }


    ImGui::SameLine(
        0.0f,
        gap
    );


    if (
        ImGui::Button(
            "Batch Processing",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        mode =
            ProcessMode::Batch;


        if (modeChange)
        {
            modeChange(mode);
        }
    }
}


void ProcessPanel::drawSingle()
{
    if (singlePreview.valid())
    {
        drawPreview(
            singlePreview
        );

        return;
    }


    drawSingleEmpty();
}


void ProcessPanel::drawBatch()
{
    if (batchPreview.valid())
    {
        drawPreview(
            batchPreview
        );

        return;
    }


    drawBatchEmpty();
}


void ProcessPanel::drawPreview(
    const UiImage& image
)
{
    ImVec2 avail =
        ImGui::GetContentRegionAvail();


    avail.x =
        std::max(
            0.0f,
            avail.x - 24.0f
        );

    avail.y =
        std::max(
            0.0f,
            avail.y - 42.0f
        );


    const ImVec2 drawSize =
        fitImage(
            image.width,
            image.height,
            avail.x,
            avail.y
        );


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                ImGui::GetContentRegionAvail().x
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
        12.0f
    );


    drawTexture(
        image.textureId,
        drawSize
    );


    if (!image.name.empty())
    {
        ImGui::Spacing();

        ImGui::TextWrapped(
            "%s",
            image.name.c_str()
        );
    }
}


void ProcessPanel::drawSingleEmpty()
{
    const ImVec2 avail =
        ImGui::GetContentRegionAvail();


    constexpr float buttonWidth =
        150.0f;

    constexpr float buttonHeight =
        42.0f;


    const char* hint =
        "Or drag an image here";


    const ImVec2 hintSize =
        ImGui::CalcTextSize(hint);


    const float totalHeight =
        buttonHeight
        +
        12.0f
        +
        hintSize.y;


    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY()
        +
        std::max(
            0.0f,
            (
                avail.y
                -
                totalHeight
            )
            *
            0.5f
        )
    );


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                avail.x
                -
                buttonWidth
            )
            *
            0.5f
        )
    );


    if (
        ImGui::Button(
            "Select Image",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (pickSingle)
        {
            pickSingle();
        }
    }


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                avail.x
                -
                hintSize.x
            )
            *
            0.5f
        )
    );


    ImGui::TextDisabled(
        "%s",
        hint
    );
}


void ProcessPanel::drawBatchEmpty()
{
    const ImVec2 avail =
        ImGui::GetContentRegionAvail();


    constexpr float buttonWidth =
        150.0f;

    constexpr float buttonHeight =
        42.0f;


    const char* hint =
        "Select a folder containing images";


    const ImVec2 hintSize =
        ImGui::CalcTextSize(hint);


    const float totalHeight =
        buttonHeight
        +
        12.0f
        +
        hintSize.y;


    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY()
        +
        std::max(
            0.0f,
            (
                avail.y
                -
                totalHeight
            )
            *
            0.5f
        )
    );


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                avail.x
                -
                buttonWidth
            )
            *
            0.5f
        )
    );


    if (
        ImGui::Button(
            "Select Folder",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (pickFolder)
        {
            pickFolder();
        }
    }


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                avail.x
                -
                hintSize.x
            )
            *
            0.5f
        )
    );


    ImGui::TextDisabled(
        "%s",
        hint
    );
}


void ProcessPanel::drawSingleButtons()
{
    constexpr float buttonWidth =
        100.0f;

    constexpr float buttonHeight =
        36.0f;

    constexpr float gap =
        12.0f;


    const float totalWidth =
        buttonWidth
        *
        2.0f
        +
        gap;


    const float availableWidth =
        ImGui::GetContentRegionAvail().x;


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                availableWidth
                -
                totalWidth
            )
            *
            0.5f
        )
    );

    if (
        ImGui::Button(
            "New",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (newCategory)
        {
            newCategory();
        }
    }


    ImGui::SameLine(
        0.0f,
        gap
    );


    if (
        ImGui::Button(
            "Confirm",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (singleConfirm)
        {
            singleConfirm();
        }
    }
}


void ProcessPanel::drawBatchButtons()
{
    constexpr float buttonWidth =
        82.0f;

    constexpr float buttonHeight =
        36.0f;

    constexpr float gap =
        20.0f;


    const float totalWidth =
        buttonWidth
        *
        4.0f
        +
        gap
        *
        3.0f;


    const float availableWidth =
        ImGui::GetContentRegionAvail().x;


    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX()
        +
        std::max(
            0.0f,
            (
                availableWidth
                -
                totalWidth
            )
            *
            0.5f
        )
    );


    if (
        ImGui::Button(
            "Previous",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (batchPrev)
        {
            batchPrev();
        }
    }


    ImGui::SameLine(
        0.0f,
        gap
    );


    if (
        ImGui::Button(
            "New",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (newCategory)
        {
            newCategory();
        }
    }


    ImGui::SameLine(
        0.0f,
        gap
    );


    if (
        ImGui::Button(
            "Confirm",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (batchConfirm)
        {
            batchConfirm();
        }
    }


    ImGui::SameLine(
        0.0f,
        gap
    );


    if (
        ImGui::Button(
            "Next",
            ImVec2(
                buttonWidth,
                buttonHeight
            )
        )
    )
    {
        if (batchNext)
        {
            batchNext();
        }
    }
}
