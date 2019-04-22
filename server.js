const express=require('express');
const bodyParser= require ('body-parser');
const hbs=require('hbs');
const app=express();
const multer=require('multer');
const predict = require('./predict_photo.js')
const fs=require('fs');



app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json());

app.use('/',express.static('public'));

app.use(function(err, req, res, next){
  //set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err:{};

  //render the error page
  res.status(err.status || 500);
  res.render('error');
});

app.set('view engine','hbs' );
app.set('views','views');

const multerConf = {
  storage: multer.diskStorage({
    destination : function(req, file, next){
      next(null, './public/uploads');
    },
    filename: function(req, file, next){
      const ext = file.mimetype.split('/')[1];
      const fileName=file.fieldname + '-' +Date.now()+ '.' +ext;

      next(null,fileName);
      //console.log(file);

    }
  })
};


app.get('/upload_camera', function(req,res,next){
  res.render('page2');
})

app.get('/upload_local', function(req,res,next){
  res.render('page3');
})


app.post('/upload', multer(multerConf).single('myImage'), function(req,res){

     console.log(req.file);
     // read binary data
   var bitmap = fs.readFileSync(req.file.path);
   // convert binary data to base64 encoded string
     var imgurl=new Buffer(bitmap).toString('base64');
     res.send({'imgurl': imgurl})
  }
);




app.listen(7171,function(){
  console.log('server running on port 7171');
})
