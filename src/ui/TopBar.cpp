#include "ui/TopBar.h"

#include <algorithm>
#include <string>
#include <utility>

#include "imgui.h"


void TopBar::setGroupCount(
    int count
)
{
    groupCount =
        std::max(
            0,
            count
        );


    if (groupCount == 0)
    {
        activeGroup = 0;

        return;
    }


    activeGroup =
        std::clamp(
            activeGroup,
            0,
            groupCount - 1
        );
}


void TopBar::setActiveGroup(
    int index
)
{
    if (groupCount <= 0)
    {
        activeGroup = 0;

        return;
    }


    activeGroup =
        std::clamp(
            index,
            0,
            groupCount - 1
        );
}


int TopBar::getActiveGroup() const
{
    return activeGroup;
}


void TopBar::setRootInfo(
    const std::string& newRootName,
    const std::string& newRootPath
)
{
    rootName =
        newRootName;


    rootPath =
        newRootPath;


    hasRoot =
        !rootPath.empty();
}


void TopBar::clearRoot()
{
    rootName.clear();

    rootPath.clear();


    hasRoot = false;


    groupCount = 0;

    activeGroup = 0;
}


void TopBar::setSelectFolder(
    Action callback
)
{
    selectFolder =
        std::move(callback);
}


void TopBar::setReselectFolder(
    Action callback
)
{
    reselectFolder =
        std::move(callback);
}


void TopBar::setGroupClick(
    GroupClick callback
)
{
    groupClick =
        std::move(callback);
}


void TopBar::draw()
{
    // ==========================================
    // 还没有选择 Group Root
    // ==========================================

    if (!hasRoot)
    {
        if (
            ImGui::Button(
                "Select Group Folder",
                ImVec2(
                    190.0f,
                    38.0f
                )
            )
        )
        {
            if (selectFolder)
            {
                selectFolder();
            }
        }


        ImGui::SameLine(
            0.0f,
            14.0f
        );


        ImGui::TextDisabled(
            "Select a folder containing your groups."
        );


        return;
    }


    // ==========================================
    // Root 信息
    // ==========================================

    ImGui::TextDisabled(
        "Selected:"
    );


    ImGui::SameLine();


    ImGui::Text(
        "%s",
        rootName.c_str()
    );


    // Hover 时显示完整路径
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "%s",
            rootPath.c_str()
        );
    }


    ImGui::SameLine(
        0.0f,
        24.0f
    );


    ImGui::Text(
        "Groups: %d",
        groupCount
    );


    ImGui::SameLine(
        0.0f,
        24.0f
    );


    if (
        ImGui::Button(
            "Reselect",
            ImVec2(
                92.0f,
                30.0f
            )
        )
    )
    {
        if (reselectFolder)
        {
            reselectFolder();
        }
    }


    ImGui::Spacing();


    // ==========================================
    // Group Buttons
    // ==========================================

    constexpr float barHeight =
        52.0f;

    constexpr float buttonWidth =
        48.0f;

    constexpr float buttonHeight =
        34.0f;


    ImGui::PushStyleVar(
        ImGuiStyleVar_ItemSpacing,
        ImVec2(
            12.0f,
            8.0f
        )
    );


    ImGui::BeginChild(
        "##groupBar",
        ImVec2(
            0.0f,
            barHeight
        ),
        ImGuiChildFlags_None,
        ImGuiWindowFlags_HorizontalScrollbar
    );


    if (groupCount == 0)
    {
        ImGui::TextDisabled(
            "No group folders found."
        );
    }


    for (
        int i = 0;
        i < groupCount;
        ++i
    )
    {
        if (i > 0)
        {
            ImGui::SameLine();
        }


        ImGui::PushID(i);


        const bool selected =
            i == activeGroup;


        if (selected)
        {
            ImGui::PushStyleColor(
                ImGuiCol_Button,
                ImGui::GetStyleColorVec4(
                    ImGuiCol_ButtonActive
                )
            );
        }


        const std::string label =
            std::to_string(
                i + 1
            );


        const bool clicked =
            ImGui::Button(
                label.c_str(),
                ImVec2(
                    buttonWidth,
                    buttonHeight
                )
            );


        if (selected)
        {
            ImGui::PopStyleColor();
        }


        if (clicked)
        {
            activeGroup = i;


            if (groupClick)
            {
                groupClick(i);
            }
        }


        ImGui::PopID();
    }


    ImGui::EndChild();


    ImGui::PopStyleVar();
}
