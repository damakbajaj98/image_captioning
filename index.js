
$('#webcamm').on('click',function(){
  $.get({url:'/upload_camera',success:function(){
    window.location='/upload_camera';
  }
})
});


$('#local').on('click',function(){
  $.get({url:'/upload_local',success:function(){
    window.location='/upload_local';
  }
})
});
