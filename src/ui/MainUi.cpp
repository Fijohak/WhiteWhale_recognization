#include "app/GroupManager.h"
#include "tools/FolderDialog.h"
#include "ui/MainUi.h"

#include <algorithm>
#include <utility>

#include "imgui.h"

MainUi::MainUi()
{

    topBar.setSelectFolder(
        [this]()
        {
            if (
                events.onSelectGroupFolder)
            {
                events.onSelectGroupFolder();
            }
        });

    topBar.setReselectFolder(
        [this]()
        {
            if (
                events.onReselectGroupFolder)
            {
                events.onReselectGroupFolder();
            }
        });

    topBar.setGroupClick(
        [this](int index)
        {
            activeGroup = index;

            if (events.onGroupClick)
            {
                events.onGroupClick(
                    index);
            }
        });

    comparePanel.setImageClick(
        [this](int imageIndex)
        {
            if (events.onImageClick)
            {
                events.onImageClick(
                    activeGroup,
                    imageIndex);
            }
        });

    processPanel.setModeChange(
        [this](ProcessMode mode)
        {
            if (events.onModeChange)
            {
                events.onModeChange(
                    mode);
            }
        });

    processPanel.setPickSingle(
        [this]()
        {
            if (events.onPickSingle)
            {
                events.onPickSingle();
            }
        });

    processPanel.setSingleDrop(
        [this](
            const std::string &path)
        {
            if (events.onSingleDrop)
            {
                events.onSingleDrop(
                    path);
            }
        });

    processPanel.setSingleConfirm(
        [this]()
        {
            if (events.onSingleConfirm)
            {
                events.onSingleConfirm();
            }
        });

    processPanel.setPickFolder(
        [this]()
        {
            if (events.onPickFolder)
            {
                events.onPickFolder();
            }
        });

    processPanel.setBatchPrev(
        [this]()
        {
            if (events.onBatchPrev)
            {
                events.onBatchPrev();
            }
        });

    processPanel.setBatchConfirm(
        [this]()
        {
            if (events.onBatchConfirm)
            {
                events.onBatchConfirm();
            }
        });

    processPanel.setBatchNext(
        [this]()
        {
            if (events.onBatchNext)
            {
                events.onBatchNext();
            }
        });

    processPanel.setNewCategory(
        [this]()
        {
            if (events.onNewCategory)
            {
                events.onNewCategory();
            }
        }
    );
}

void MainUi::setEvents(
    UiEvents newEvents)
{
    events =
        std::move(newEvents);
}

void MainUi::setGroupCount(
    int count)
{
    topBar.setGroupCount(
        count);
}

void MainUi::setActiveGroup(
    int index)
{
    activeGroup = index;

    topBar.setActiveGroup(
        index);
}

int MainUi::getActiveGroup() const
{
    return activeGroup;
}

void MainUi::setCompareImages(
    const std::vector<UiImage> &images)
{
    comparePanel.setImages(
        images);
}

void MainUi::clearCompareImages()
{
    comparePanel.clearImages();
}

void MainUi::setSinglePreview(
    const UiImage &image)
{
    processPanel.setSinglePreview(
        image);
}

void MainUi::clearSinglePreview()
{
    processPanel.clearSinglePreview();
}

void MainUi::setBatchPreview(
    const UiImage &image)
{
    processPanel.setBatchPreview(
        image);
}

void MainUi::clearBatchPreview()
{
    processPanel.clearBatchPreview();
}

void MainUi::setProcessMode(
    ProcessMode mode)
{
    processPanel.setMode(
        mode);
}

ProcessMode MainUi::getProcessMode() const
{
    return processPanel.getMode();
}

void MainUi::resetProcessMode()
{
    processPanel.resetMode();
}

void MainUi::handleFileDrop(
    const std::string &path)
{
    processPanel.handleFileDrop(
        path);
}

void MainUi::draw()
{
    ImGuiViewport *viewport =
        ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(
        viewport->WorkPos);

    ImGui::SetNextWindowSize(
        viewport->WorkSize);

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowRounding,
        0.0f);

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowBorderSize,
        0.0f);

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin(
        "##mainUi",
        nullptr,
        flags);

    // =========================================
    // 顶部组别区域
    // =========================================

    topBar.draw();

    ImGui::Separator();

    // =========================================
    // 左右主体
    // =========================================

    const ImVec2 avail =
        ImGui::GetContentRegionAvail();

    constexpr float gap =
        10.0f;

    float leftWidth =
        avail.x * 0.50f;

    leftWidth =
        std::max(
            280.0f,
            leftWidth);

    if (
        leftWidth >
        avail.x - 260.0f)
    {
        leftWidth =
            std::max(
                200.0f,
                avail.x - 260.0f);
    }

    // =========================================
    // 左侧
    // =========================================

    ImGui::BeginChild(
        "##comparePane",
        ImVec2(
            leftWidth,
            0.0f),
        ImGuiChildFlags_Borders,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);

    comparePanel.draw();

    ImGui::EndChild();

    ImGui::SameLine(
        0.0f,
        gap);

    // =========================================
    // 右侧
    // =========================================

    ImGui::BeginChild(
        "##processPane",
        ImVec2(
            0.0f,
            0.0f),
        ImGuiChildFlags_Borders);

    processPanel.draw();

    ImGui::EndChild();

    ImGui::End();

    ImGui::PopStyleVar(2);
}


void MainUi::setGroupRoot(
    const std::string& rootName,
    const std::string& rootPath,
    int groupCount
)
{
    topBar.setRootInfo(
        rootName,
        rootPath
    );


    topBar.setGroupCount(
        groupCount
    );


    activeGroup = 0;


    topBar.setActiveGroup(
        0
    );


    // 换了整个 Group Root，
    // 原来左边显示的图片已经失效。
    comparePanel.clearImages();
}


void MainUi::clearGroupRoot()
{
    topBar.clearRoot();

    activeGroup = 0;

    comparePanel.clearImages();
}
