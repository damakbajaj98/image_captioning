const express=require('express');
const bodyParser= require ('body-parser');
const app=express();
const fs= require('fs')
const hbs=require('hbs')

app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));
app.use(bodyParser.json({limit: '50mb', extended: true}));

app.use('/',express.static('public'));


app.set('view engine','hbs' );
app.set('views','views');

app.get('/upload_camera', function(req,res,next){
  res.render('index2');
})

app.get('/upload_local', function(req,res,next){
  res.render('index3');
})



app.post('/predict',function(req,res){
  image=req.body.img
  // console.log(req.body.img)
  fs.writeFile("myfile.txt",image,function(){
    console.log('saved')
  })
  var spawn = require("child_process").spawn;
  var process = spawn('python',["./img_cap_model.py"] );

  process.stdout.on('data', function(data) {
      console.log(data.toString())
      res.json({data:data.toString()});
  });
});




app.listen(7171,function(){
  console.log('server running on port 7171');
})
