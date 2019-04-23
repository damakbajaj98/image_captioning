let myimage

function encodeImageFileAsURL() {

  var filesSelected = document.getElementById("image-upload").files;
  if (filesSelected.length > 0) {
    var fileToLoad = filesSelected[0];

    var fileReader = new FileReader();

    fileReader.onload = function(fileLoadedEvent) {
      var srcData = fileLoadedEvent.target.result; // <--- data: base64

      var newImage = document.createElement('img');
      newImage.src = srcData;

      document.getElementById("js--image-preview").innerHTML = newImage.outerHTML;
      // alert("Converted Base64 version is " + document.getElementById("imgTest").innerHTML);
      console.log(document.getElementById("js--image-preview").innerHTML);
      console.log("this is the base64")
      console.log(srcData);
      myimage=srcData
    }
    fileReader.readAsDataURL(fileToLoad);
  }
}


function playaudio(){
  console.log('playing audio')
  $('#audiocontainer').empty();
  $('#audiocontainer').append(`<audio controls>
    <source src="captionaudio.mp3" type="audio/mp3">

    <p>Your browser doesn't support HTML5 audio. Here is a <a href="viper.mp3">link to the audio</a> instead.</p>
  </audio>`)
}

$("#generate").on('click',function(){
  $.post({url:'/predict',data:{img:myimage},success:function(res){
    console.log(res)
    $('#caption').empty();
    $('#caption').append(`<p id="cap">${res.data}</p>`)
    playaudio();
  }
})
})
