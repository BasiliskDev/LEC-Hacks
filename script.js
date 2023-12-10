
var display = document.getElementById('img_display');
var input = document.getElementById('img_input');

input.onchange = function() {
    var reader = new FileReader();
    reader.onload = function(){
        var dataURL = reader.result;
        display.src = dataURL;
    }
    reader.readAsDataURL(input.files[0]);
}