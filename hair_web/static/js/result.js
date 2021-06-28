  var intervalId;
  window.addEventListener("load", function(event) {
      intervalId = setInterval(checkImage,1000);
  });

  function checkImage(){
      var img = new Image();
      img.src = "{{ url_for('static', filename='images/12345/result.png') }}";
      img.onload = function(){
          document.getElementById("changeImg").setAttribute("src","{{ url_for('static', filename='images/12345/result.png') }}")
          document.getElementById("spinner-section").setAttribute("style","visibility:collapse");
          clearInterval(intervalId);
      }
}