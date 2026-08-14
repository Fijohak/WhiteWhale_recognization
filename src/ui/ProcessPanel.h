#pragma once

#include <functional>
#include <string>

#include "ui/UiImage.h"


enum class ProcessMode
{
    Select,

    Single,

    Batch
};


class ProcessPanel
{
public:

    using Action =
        std::function<void()>;


    using ModeChange =
        std::function<void(ProcessMode)>;


    using FileDrop =
        std::function<void(
            const std::string&
        )>;


public:

    void draw();


    void setMode(
        ProcessMode mode
    );


    ProcessMode getMode() const;


    void resetMode();


    void setSinglePreview(
        const UiImage& image
    );


    void clearSinglePreview();


    void setBatchPreview(
        const UiImage& image
    );


    void clearBatchPreview();


    void handleFileDrop(
        const std::string& path
    );


    void setModeChange(
        ModeChange callback
    );


    void setPickSingle(
        Action callback
    );


    void setSingleDrop(
        FileDrop callback
    );


    void setSingleConfirm(
        Action callback
    );


    void setPickFolder(
        Action callback
    );


    void setBatchPrev(
        Action callback
    );


    void setBatchConfirm(
        Action callback
    );

    void setNewCategory(
        Action callback
    );


    void setBatchNext(
        Action callback
    );


private:

    void drawModeSelect();

    void drawSingle();

    void drawBatch();


    void drawPreview(
        const UiImage& image
    );


    void drawSingleEmpty();

    void drawBatchEmpty();


    void drawSingleButtons();

    void drawBatchButtons();

    void drawBackButton();


private:

    ProcessMode mode =
        ProcessMode::Select;


    UiImage singlePreview;

    UiImage batchPreview;


    ModeChange modeChange;

    Action pickSingle;

    FileDrop singleDrop;

    Action singleConfirm;

    Action pickFolder;

    Action batchPrev;

    Action batchConfirm;

    Action newCategory;

    Action batchNext;
};
