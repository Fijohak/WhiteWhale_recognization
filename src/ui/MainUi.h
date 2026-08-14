#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ui/ComparePanel.h"
#include "ui/ProcessPanel.h"
#include "ui/TopBar.h"


struct UiEvents
{
    std::function<void()>
        onSelectGroupFolder;


    std::function<void()>
        onReselectGroupFolder;


    std::function<void(int)>
        onGroupClick;


    std::function<void(int, int)>
        onImageClick;


    std::function<void(ProcessMode)>
        onModeChange;


    std::function<void()>
        onPickSingle;


    std::function<void(const std::string&)>
        onSingleDrop;


    std::function<void()>
        onSingleConfirm;


    std::function<void()>
        onPickFolder;


    std::function<void()>
        onNewCategory;


    std::function<void()>
        onBatchPrev;


    std::function<void()>
        onBatchConfirm;


    std::function<void()>
        onBatchNext;
};

class MainUi
{
public:

    MainUi();


    void draw();


    void setEvents(
        UiEvents events
    );


    void setGroupCount(
        int count
    );


    void setActiveGroup(
        int index
    );


    int getActiveGroup() const;


    void setCompareImages(
        const std::vector<UiImage>& images
    );


    void clearCompareImages();


    void setSinglePreview(
        const UiImage& image
    );


    void clearSinglePreview();


    void setBatchPreview(
        const UiImage& image
    );


    void clearBatchPreview();


    void setProcessMode(
        ProcessMode mode
    );


    ProcessMode getProcessMode() const;


    void resetProcessMode();


    void handleFileDrop(
        const std::string& path
    );

    void setGroupRoot(
        const std::string& rootName,
        const std::string& rootPath,
        int groupCount
    );


    void clearGroupRoot();

private:

    TopBar topBar;

    ComparePanel comparePanel;

    ProcessPanel processPanel;


    UiEvents events;


    int activeGroup = 0;
};
