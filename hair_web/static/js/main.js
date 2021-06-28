function setSourceThumbnail(event) {
     var reader = new FileReader(); 
     reader.onload = function(event) 
     { 
         var img = document.getElementById("source-jpg-img");
         img.setAttribute("src", event.target.result);     
         img.setAttribute("style","opacity :1.0")
         document.getElementById("photo-text").innerText = "";
     };
      reader.readAsDataURL(event.target.files[0]);
}

function setReferenceThumbnail(event){
    var reader = new FileReader(); 
    reader.onload = function(event) 
    { 
        var img = document.getElementById("reference-jpg-img");
        img.setAttribute("src", event.target.result);     
        img.setAttribute("style","opacity :1.0")
        document.getElementById("photo-text").innerText = "";
    };
     reader.readAsDataURL(event.target.files[0]);
}

var typeSelect = document.getElementById("type-select");
typeSelect.addEventListener('change',(event)=>{
    var option_value = event.target.value;
    var reference_text = document.getElementById("reference-text");
    if(option_value == 1){
        reference_text.innerHTML = "바꾸고 싶은 헤어스타일을 업로드해 주세요";
        document.getElementById("select-style").style.visibility = "visible";
        document.getElementById("color-input").style.visibility= "collapse";
    }
    else{
        reference_text.innerHTML = "바꾸고 싶은 머리 색상을 업로드해 주세요";
        document.getElementById("select-style").style.visibility = "collapse";
        document.getElementById("color-input").style.visibility= "visible";
    }
});