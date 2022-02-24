//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

var fileSource = null;

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
    // prevent default behaviour
    e.preventDefault();
    e.stopPropagation();

    fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
    // handle file selecting
    var files = e.target.files || e.dataTransfer.files;
    fileDragHover(e);
    for (var i = 0, f; f = files[i]; i++) {
        previewFile(f);
        fileGetSrc(f);
    }
}

function fileGetSrc(file) {
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        fileSource = reader.result;
    };
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var uploadCaption = document.getElementById("upload-caption");

//========================================================================
// Main button events
//========================================================================

function clearImage() {
    // reset selected files
    fileSelect.value = "";

    // remove image sources and hide them
    imagePreview.src = "";

    hide(imagePreview);
    show(uploadCaption);

    clearPredictions();
    
    fileSource = null;
}

function previewFile(file) {
    // show the preview of the image
    console.log(file.name);
    var fileName = encodeURI(file.name);

    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        imagePreview.src = URL.createObjectURL(file);

        show(imagePreview);
        hide(uploadCaption);

        clearPredictions();
    };
}

//========================================================================
// Helper functions
//========================================================================

function clearPredictions() {
    prediction_imgs = document.getElementsByClassName('prediction_img');
    for (pred_img of prediction_imgs) {
        // pred_img.src = '';
        visible_not(pred_img);
    }
    prediction_divs = document.getElementsByClassName('prediction_text');
    for (pred_div of prediction_divs) {
        // pred_div.innerHTML = '';
        visible_not(pred_div);
    }
}

function predictImage(modelName, id) {
    if (fileSource == null) {
        return;
    }
    let prediction = document.getElementById(id);
    let prediction_img = prediction.getElementsByClassName('prediction_img')[0];
    let prediction_div = prediction.getElementsByClassName('prediction_text')[0];
    let prediction_loader = prediction.getElementsByClassName('loader')[0];
    visible_not(prediction_img);
    visible_not(prediction_div);
    visible(prediction_loader);

    if (fileSource == null) {
        return
    }
    fetch(`/predictwithheatmap/${modelName}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(fileSource)
        })
        .then(resp => {
            if (resp.ok)
                resp.json().then(data => {
                    prediction_div.innerHTML = `${data.class_name} / Probability: ${data.probability}`;
                    prediction_img.src = data.img_with_heatmap;
                    visible(prediction_img);
                    visible(prediction_div);
                    visible_not(prediction_loader);
                });
        })
        .catch(err => {
            console.log("An error occured", err.message);
            window.alert("Oops! Something went wrong.");
        });
}

function displayImage(image, id) {
    // display image on given id <img> element
    let display = document.getElementById(id);
    display.src = image;
    show(display);
}

function hide(el) {
    // hide an element
    el.classList.add("hidden");
}

function show(el) {
    // show an element
    el.classList.remove("hidden");
}

function visible_not(el) {
    // hide an element
    el.classList.add("not_visible");
}

function visible(el) {
    // show an element
    el.classList.remove("not_visible");
}
