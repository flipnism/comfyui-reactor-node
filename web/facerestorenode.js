import { app } from "../../scripts/app.js";
const ActivateNodeType = "RestoreAndScale"
const id = "Reactor.FaceRestoreSwap";
function set_preview_not_editable(node) {
    if (node.type == ActivateNodeType) {
        for (var w of node.widgets) {
            if (w.name == "preview") {
                w.inputEl.readOnly = true;
                return w;
            }
        }
    }
    return null;
}

function update_preview(node) {
    if (node.type != ActivateNodeType)
        return "";
    var dict = new Object();

    for (var w of node.widgets)
        dict[w.name] = w;
    dict.scale_restore.readOnly = true;
    console.log(dict.scale_restore.readOnly)
}

function customize_node(node) {
    if ('upscale_model_name' in node)
        return;
    var preview_widget = set_preview_not_editable(node);
    if (preview_widget) {
        node.preview_widget = preview_widget;
        update_preview(node)
    }
}

app.registerExtension({
    name: id,

    init() {





    },
    loadedGraphNode(node, app) {
        customize_node(node);
    },
    nodeCreated(node, app) {
        function checkTextArea(event) {
            update_preview(node);
        }

        for (var i in node.widgets) {
            var w = node.widgets[i];
            if (w.type == "customtext" && 'inputEl' in w) {
                w.inputEl.addEventListener("input", checkTextArea.bind(node));
            }
        }

        node.onWidgetChanged = function (name, value, old_value) {
            update_preview(node);
        }
    }
});