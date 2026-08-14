#pragma once

#include <functional>
#include <string>


class TopBar
{
public:

    using Action =
        std::function<void()>;


    using GroupClick =
        std::function<void(int)>;


public:

    void draw();


    void setGroupCount(
        int count
    );


    void setActiveGroup(
        int index
    );


    int getActiveGroup() const;


    void setRootInfo(
        const std::string& rootName,
        const std::string& rootPath
    );


    void clearRoot();


    void setSelectFolder(
        Action callback
    );


    void setReselectFolder(
        Action callback
    );


    void setGroupClick(
        GroupClick callback
    );


private:

    int groupCount = 0;

    int activeGroup = 0;


    bool hasRoot = false;


    std::string rootName;

    std::string rootPath;


    Action selectFolder;

    Action reselectFolder;

    GroupClick groupClick;
};
