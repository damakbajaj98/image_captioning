
$('#capture').on('click',function(){
  $.post({url:'/upload',success:function(data){
    x=data.imgurl
  }
})
});
