package com.tencent.yolov8ncnn.Test;

import androidx.appcompat.app.AppCompatActivity;


import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;
// ContentResolver dependency

import com.google.mediapipe.formats.proto.LandmarkProto.Landmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutioncore.VideoInput;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;
import com.tencent.yolov8ncnn.R;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class TestActivity extends AppCompatActivity {

    public String detect_res="";
    private Handler uiHandler = new Handler();
    private static final String TAG = "MainActivity";

    private Hands hands;
    // Run the pipeline and the model inference on GPU or CPU.
    private static final boolean RUN_ON_GPU = true;

    private enum InputSource {
        UNKNOWN,
        IMAGE,
        VIDEO,
        CAMERA,
    }
    private InputSource inputSource = InputSource.UNKNOWN;

    // Image demo UI and image loader components.
    private ActivityResultLauncher<Intent> imageGetter;
    private HandsResultImageView imageView;
    // Video demo UI and video loader components.
    private VideoInput videoInput;
    private ActivityResultLauncher<Intent> videoGetter;
    // Live camera demo UI and camera components.
    private CameraInput cameraInput;

    private SolutionGlSurfaceView<HandsResult> glSurfaceView;

    private Button button_test_camera;
    private ImageView img_love;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);
//        button_test_camera=findViewById(R.id.button_test_camera);
//        System.out.println("----"+button_test_camera);
//        button_test_camera.setOnClickListener(new View.OnClickListener() {
//                                                  @Override
//                                                  public void onClick(View v) {
////                Intent intent = new Intent(MainActivity.this,CameraActivity.class);
////                startActivity(intent);
//                                                  }
//                                              }
//        );
//
//        img_love=findViewById(R.id.img_love);
        setupStaticImageDemoUiComponents();
        setupLiveDemoUiComponents();
    }

    /** Sets up core workflow for static image mode. */
    private void setupStaticImageModePipeline() {
        this.inputSource = InputSource.IMAGE;
        // Initializes a new MediaPipe Hands solution instance in the static image mode.
        hands =
                new Hands(
                        this,
                        HandsOptions.builder()
                                .setStaticImageMode(true)
                                .setMaxNumHands(2)
                                .setRunOnGpu(RUN_ON_GPU)
                                .build());

        // 将 MediaPipe Hands 解决方案连接到用户定义的 HandsResultImageView。
        /*
            也就是hand检测的结果会传到这里面来，执行hand.sent()
            类似于触发事件监听器，然后就会执行这个lambda表达式
         */
        hands.setResultListener(
                handsResult -> {
                    logWristLandmark(handsResult, /*showPixelValues=*/ true);
                    imageView.setHandsResult(handsResult);
                    runOnUiThread(() -> imageView.update());
                });
        hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

        // Updates the preview layout.
        FrameLayout frameLayout = (FrameLayout) this.findViewById(R.id.preview);
        frameLayout.removeAllViewsInLayout();
        imageView.setImageDrawable(null);
        frameLayout.addView(imageView);
        imageView.setVisibility(View.VISIBLE);
    }

    private Bitmap downscaleBitmap(Bitmap originalBitmap) {
        double aspectRatio = (double) originalBitmap.getWidth() / originalBitmap.getHeight();
        int width = imageView.getWidth();
        int height = imageView.getHeight();
        if (((double) imageView.getWidth() / imageView.getHeight()) > aspectRatio) {
            width = (int) (height * aspectRatio);
        } else {
            height = (int) (width / aspectRatio);
        }
        return Bitmap.createScaledBitmap(originalBitmap, width, height, false);
    }

    private Bitmap rotateBitmap(Bitmap inputBitmap, InputStream imageData) throws IOException {
        int orientation =
                new ExifInterface(imageData)
                        .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        if (orientation == ExifInterface.ORIENTATION_NORMAL) {
            return inputBitmap;
        }
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
            default:
                matrix.postRotate(0);
        }
        return Bitmap.createBitmap(
                inputBitmap, 0, 0, inputBitmap.getWidth(), inputBitmap.getHeight(), matrix, true);
    }

    /** Sets up the UI components for the static image demo. */
    private void setupStaticImageDemoUiComponents() {
        // The Intent to access gallery and read images as bitmap.
        imageGetter =
                registerForActivityResult(
                        new ActivityResultContracts.StartActivityForResult(),
                        result -> {
                            Intent resultIntent = result.getData();
                            if (resultIntent != null) {
                                if (result.getResultCode() == RESULT_OK) {
                                    Bitmap bitmap = null;
                                    try {
                                        bitmap =
                                                downscaleBitmap(
                                                        MediaStore.Images.Media.getBitmap(
                                                                this.getContentResolver(), resultIntent.getData()));
                                    } catch (IOException e) {
                                        Log.e(TAG, "Bitmap reading error:" + e);
                                    }
                                    try {
                                        InputStream imageData =
                                                this.getContentResolver().openInputStream(resultIntent.getData());
                                        bitmap = rotateBitmap(bitmap, imageData);
                                    } catch (IOException e) {
                                        Log.e(TAG, "Bitmap rotation error:" + e);
                                    }
                                    if (bitmap != null) {
                                        //这里获取图片
                                        System.out.println(bitmap);
//                                        save_image_permisstion();
//                                        saveBitmap(bitmap,"test");
//                                        这里图片手势检测
                                        hands.send(bitmap);
                                    }
                                }
                            }
                        });
//        Button loadImageButton = findViewById(R.id.button_load_picture);
//        loadImageButton.setOnClickListener(
//                v -> {
//                    // 鼠标点击事件的时候跳出框去本地选择图片
//                    if (inputSource != InputSource.IMAGE) {
//                        stopCurrentPipeline();
//                        setupStaticImageModePipeline();
//                    }
//                    // Reads images from gallery.
//                    Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
//                    //选择完成图片以后接收选择的图片数据
//                    pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
////                    执行registerForActivityResult 里面调用了手势检测hand.sent()
//                    imageGetter.launch(pickImageIntent);
//                });
//        // 手势检测的结果显示到当前页面上
//        imageView = new HandsResultImageView(this);
    }

    /** Sets up the UI components for the live demo with camera input. */
    private void setupLiveDemoUiComponents() {
        //打开相机
        Button startCameraButton = findViewById(R.id.start_camera);
        startCameraButton.setOnClickListener(
                v -> {
                    if (inputSource == InputSource.CAMERA) {
                        return;
                    }
                    stopCurrentPipeline();
                    setupStreamingModePipeline(InputSource.CAMERA);
                });
    }

    /** Sets up core workflow for streaming mode. */
    private void setupStreamingModePipeline(InputSource inputSource) {
        this.inputSource = inputSource;
        // Initializes a new MediaPipe Hands solution instance in the streaming mode.
//        创建手部检测对象
        hands =
                new Hands(
                        this,
                        HandsOptions.builder()
                                .setStaticImageMode(false)
                                .setMaxNumHands(2)
                                .setRunOnGpu(RUN_ON_GPU)
                                .build());
//        添加错误监听器
        hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

        if (inputSource == InputSource.CAMERA) {
//            监听当前页面的视频输入流
            cameraInput = new CameraInput(this);
//            这个不太明白
            cameraInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
        } else if (inputSource == InputSource.VIDEO) {
            videoInput = new VideoInput(this);
            videoInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
        }

        // 使用用户定义的 HandsResultGlRenderer 初始化新的 Gl 表面视图。
        glSurfaceView =
                new SolutionGlSurfaceView<>(this, hands.getGlContext(), hands.getGlMajorVersion());
        glSurfaceView.setSolutionResultRenderer(new HandsResultGlRenderer());
        glSurfaceView.setRenderInputImage(true);
        //回调函数，设置监听器就可以根据获取的结果进行下一步处理
        hands.setResultListener(
                handsResult -> {
                    logWristLandmark(handsResult, /*showPixelValues=*/ false);
                    glSurfaceView.setRenderData(handsResult);
                    glSurfaceView.requestRender();
                    int numHands = handsResult.multiHandLandmarks().size();
//                    特殊检测
                    if(numHands>1){
                        //检测两只手
                        detect_res=twoHandDetect(handsResult.multiHandLandmarks().get(0).getLandmarkList()
                                ,handsResult.multiHandLandmarks().get(1).getLandmarkList());
                    }else{
//                        检测一只手的
                        for (int i = 0; i < numHands; ++i) {
                            //判断是否是左手
//                            boolean isLeftHand = handsResult.multiHandedness().get(i).getLabel().equals("Left");
//                        定义了一个成员变量
                            String code=baseHandCount(handsResult.multiHandLandmarks().get(i).getLandmarkList());
                            detect_res = basePost(code);

                        }
                    }

                    //      创建一个线程
                  //  new MessageShow().start();

                });

        // 附加 gl 表面视图后启动相机的可运行程序。
        // 对于视频输入源，当视频uri可用时将调用videoInput.start()。
        if (inputSource == InputSource.CAMERA) {
            glSurfaceView.post(this::startCamera);
        }

        // Updates the preview layout.
        FrameLayout frameLayout = findViewById(R.id.preview);
     //   imageView.setVisibility(View.GONE);
        frameLayout.removeAllViewsInLayout();
        frameLayout.addView(glSurfaceView);
        glSurfaceView.setVisibility(View.VISIBLE);
        frameLayout.requestLayout();
    }

    private void startCamera() {
        cameraInput.start(
                this,
                hands.getGlContext(),
                CameraInput.CameraFacing.BACK,
                glSurfaceView.getWidth(),
                glSurfaceView.getHeight());
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (inputSource == InputSource.CAMERA) {
            // Restarts the camera and the opengl surface rendering.
            cameraInput = new CameraInput(this);
            cameraInput.setNewFrameListener(textureFrame -> hands.send(textureFrame));
            glSurfaceView.post(this::startCamera);
            glSurfaceView.setVisibility(View.VISIBLE);
        } else if (inputSource == InputSource.VIDEO) {
            videoInput.resume();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (inputSource == InputSource.CAMERA) {
            glSurfaceView.setVisibility(View.GONE);
            cameraInput.close();
        } else if (inputSource == InputSource.VIDEO) {
            videoInput.pause();
        }
    }

    private void stopCurrentPipeline() {
        if (cameraInput != null) {
            cameraInput.setNewFrameListener(null);
            cameraInput.close();
        }
        if (videoInput != null) {
            videoInput.setNewFrameListener(null);
            videoInput.close();
        }
        if (glSurfaceView != null) {
            glSurfaceView.setVisibility(View.GONE);
        }
        if (hands != null) {
            hands.close();
        }
    }


    private void logWristLandmark(HandsResult result, boolean showPixelValues) {
        if (result.multiHandLandmarks().isEmpty()) {
            return;
        }
        NormalizedLandmark wristLandmark =
                result.multiHandLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
        // For Bitmaps, show the pixel values. For texture inputs, show the normalized coordinates.
        if (showPixelValues) {
            int width = result.inputBitmap().getWidth();
            int height = result.inputBitmap().getHeight();
            Log.i(
                    TAG,
                    String.format(
                            "MediaPipe Hand wrist coordinates (pixel values): x=%f, y=%f",
                            wristLandmark.getX() * width, wristLandmark.getY() * height));
        } else {
            Log.i(
                    TAG,
                    String.format(
                            "MediaPipe Hand wrist normalized coordinates (value range: [0, 1]): x=%f, y=%f",
                            wristLandmark.getX(), wristLandmark.getY()));
        }
        if (result.multiHandWorldLandmarks().isEmpty()) {
            return;
        }
        Landmark wristWorldLandmark =
                result.multiHandWorldLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
        Log.i(
                TAG,
                String.format(
                        "MediaPipe Hand wrist world coordinates (in meters with the origin at the hand's"
                                + " approximate geometric center): x=%f m, y=%f m, z=%f m",
                        wristWorldLandmark.getX(), wristWorldLandmark.getY(), wristWorldLandmark.getZ()));
    }


/*
    保存图片方法
*/

    public void save_image_permisstion(){
        String[] PERMISSIONS = {
                "android.permission.READ_EXTERNAL_STORAGE",
                "android.permission.WRITE_EXTERNAL_STORAGE" };
        //检测是否有写的权限
        int permission = ContextCompat.checkSelfPermission(this,
                "android.permission.WRITE_EXTERNAL_STORAGE");
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // 没有写的权限，去申请写的权限，会弹出对话框
            ActivityCompat.requestPermissions(this, PERMISSIONS,1);
        }
    }
    //    保存页面view到图片
    public void SaveBitmapFromView(View view) {
        int w = view.getWidth();
        int h = view.getHeight();
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);
        view.layout(0, 0, w, h);
        view.draw(c);
        // 缩小图片
        Matrix matrix = new Matrix();
        matrix.postScale(0.5f,0.5f); //长和宽放大缩小的比例
        bmp = Bitmap.createBitmap(bmp,0,0,        bmp.getWidth(),bmp.getHeight(),matrix,true);
        DateFormat format = new SimpleDateFormat("yyyyMMddHHmmss");
        saveBitmap(bmp,format.format(new Date())+".JPEG");
    }
    /*
     * 保存文件，文件名为当前日期
     */
    public void saveBitmap(Bitmap bitmap, String bitName){
        File file = new File(this.getFilesDir(), "test.jpg");
//        if(Build.BRAND .equals("Xiaomi") ){ // 小米手机
//            fileName = Environment.getExternalStorageDirectory().getPath()+"/DCIM/Camera/"+bitName ;
//        }else{ // Meizu 、Oppo
//            fileName = Environment.getExternalStorageDirectory().getPath()+"/DCIM/"+bitName ;
//        }

        if(file.exists()){
            file.delete();
        }
        FileOutputStream out;
        try{
            out = new FileOutputStream(file);
            // 格式为 JPEG，照相机拍出的图片为JPEG格式的，PNG格式的不能显示在相册中
            if(bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out))
            {
                out.flush();
                out.close();
// 插入图库
                MediaStore.Images.Media.insertImage(this.getContentResolver(), file.getAbsolutePath(), bitName, null);
            }
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        // 发送广播，通知刷新图库的显示
    }
    //手势检测

    //    手指弯曲计算，计算五个手指那个手指是弯曲的
    private String baseHandCount(List<NormalizedLandmark> handLandmarkList){

//        从拇指开始是one，two这样
        List<Boolean> resultList=new ArrayList<>();
//        写字判断类别：
        /*
            计算拇指是否弯曲
            计算第四个点到第17点距离小于第5点到17点距离
         */
        float seventhreenx=handLandmarkList.get(17).getX();
        float seventhreeny=handLandmarkList.get(17).getY();
        float seventhreenz=handLandmarkList.get(17).getZ();
        float fourx=handLandmarkList.get(4).getX();
        float foury=handLandmarkList.get(4).getY();
        float fourz=handLandmarkList.get(4).getZ();
        float fivex=handLandmarkList.get(5).getX();
        float fivey=handLandmarkList.get(5).getY();
        float fivez=handLandmarkList.get(5).getZ();
        float distance4=
                (fourx-seventhreenx)*(fourx-seventhreenx)+
                        (foury-seventhreeny)*(foury-seventhreeny)+
                        (fourz-seventhreenz)*(fourz-seventhreenz);
        float distance5=
                (fivex-seventhreenx)*(fivex-seventhreenx)+
                        (fivey-seventhreeny)*(fivey-seventhreeny)+
                        (fivez-seventhreenz)*(fivez-seventhreenz);
        if(distance5-distance4>0){
//            有弯曲
            resultList.add(true);

        }else{
            resultList.add(false);
//            没有弯曲
        }
        /*
            计算四个手指是否弯曲
            手指的指尖距离大于关节距离
         */

        // 定义0点坐标zerox,zeroy,zeroz
        float zerox=handLandmarkList.get(0).getX();
        float zeroy=handLandmarkList.get(0).getY();
        float zeroz=handLandmarkList.get(0).getZ();

        //该遍历是:(6,8),(10,12),(14,16),(18,20)
        int[] intarr=new int[]{6,8,10,12,14,16,18,20};

        for(int i =0;i<4;i++){//0,1,2,3,,取i,i+1),2i,2i+1
            float closex = handLandmarkList.get(intarr[2*i]).getX();
            float closey = handLandmarkList.get(intarr[2*i]).getY();
            float closez = handLandmarkList.get(intarr[2*i]).getZ();
            float farx = handLandmarkList.get(intarr[2*i+1]).getX();
            float fary = handLandmarkList.get(intarr[2*i+1]).getY();
            float farz = handLandmarkList.get(intarr[2*i+1]).getZ();
            //取出点坐标与0点求欧式距离
            float close_distance=(closex-zerox)*(closex-zerox)+(closey-zeroy)*(closey-zeroy)+(closez-zeroz)*(closez-zeroz);
            float far_distance=(farx-zerox)*(farx-zerox)+
                    (fary-zeroy)*(fary-zeroy)+
                    (farz-zeroz)*(farz-zeroz);
            float between =close_distance- far_distance  ;
            float zero=0;
            if(between>zero){
                //弯曲
                resultList.add(true);
            }else{
                //  没有弯曲
                resultList.add(false);

            }
        }

        String code="";
        for(boolean b :resultList){
            if(b){
                code+="1";
            }else{
                code=code+"0";
            }
        }
        return code;
    }

    private String basePost(String code){
        String res="";
        switch (code){
            case "01111":
                res="GOOD";
//                img_love.setImageResource(R.drawable.good);
                break;
            case "10111":
                res="数字1";
//                img_love.setImageResource(R.drawable.one);
                break;
            case "11111":
                res="fist";
//                img_love.setImageResource(R.drawable.fist);
                break;


            //没有图片
            case "10011":
                res="数字2";
                break;
            case "10001":
                res="数字3";
                break;
            case "10000":
                res="数字4";
                break;
            case "00000":
                res="数字5";
                break;
            case "01110":
                res="数字6";
                break;
            case "00111":
                res="数字7";
                break;
            case "00011":
                res="数字8";
                break;
            case "11011":
                res="国际友好手势";
                break;
            case "11000":
                res="OK";
                break;
            case "00110":
                res="Love2";
                break;
            default:
                res="none";
                break;
        }

        return res;

    }
    private String twoHandDetect(List<NormalizedLandmark> landmark1,List<NormalizedLandmark> landmark2){
        String code1 = baseHandCount(landmark1);
        String code2 = baseHandCount(landmark2);
        //判断Love
        if(code1.endsWith("111")&&code2.endsWith("111")||code1.endsWith("000")&&code2.endsWith("000")){
            // 两个指尖距离小于指尖到关节的距离
            float hand1x4=landmark1.get(4).getX();
            float hand1y4=landmark1.get(4).getY();
            float hand1z4=landmark1.get(4).getZ();

            float hand2x4=landmark2.get(4).getX();
            float hand2y4=landmark2.get(4).getY();
            float hand2z4=landmark2.get(4).getZ();

            float hand1x8=landmark1.get(8).getX();
            float hand1y8=landmark1.get(8).getY();
            float hand1z8=landmark1.get(8).getZ();

            float hand2x8=landmark2.get(8).getX();
            float hand2y8=landmark2.get(8).getY();
            float hand2z8=landmark2.get(8).getZ();

            //距离计算:两只手食指指尖和拇指指尖距离计算
            float distance_muzhi=
                    (hand1x4-hand2x4)*(hand1x4-hand2x4)+
                            (hand1y4-hand2y4)*(hand1y4-hand2y4)+
                            (hand1z4-hand2z4)*(hand1z4-hand2z4);
            float distance_shizhi=
                    (hand1x8-hand2x8)*(hand1x8-hand2x8)+
                            (hand1y8-hand2y8)*(hand1y8-hand2y8)+
                            (hand1z8-hand2z8)*(hand1z8-hand2z8);
            //指尖到指尖关节距离计算
            float hand1x3=landmark1.get(3).getX();
            float hand1y3=landmark1.get(3).getY();
            float hand1z3=landmark1.get(3).getZ();
            float distance1_34=
                    (hand1x4-hand1x3)*(hand1x4-hand1x3)+
                            (hand1y4-hand1y3)*(hand1y4-hand1y3)+
                            (hand1z4-hand1z3)*(hand1z4-hand1z3);
            float hand1x7=landmark1.get(7).getX();
            float hand1y7=landmark1.get(7).getY();
            float hand1z7=landmark1.get(7).getZ();
            float distance1_78=
                    (hand1x8-hand1x7)*(hand1x8-hand1x7)+
                            (hand1y8-hand1y7)*(hand1y8-hand1y7)+
                            (hand1z8-hand1z7)*(hand1z8-hand1z7);
            float zero=0;
            if((distance1_34*1.5-distance_muzhi>zero)&&(distance1_78-distance_shizhi>zero)){
                //手势正确
                return "Love";
            }
        }
        //判断flower 五根手指都不弯曲
        if("00000".equals(code1)&&"00000".equals(code2)){
            //0点之间的距离：
            float hand1x0=landmark1.get(0).getX();
            float hand1y0=landmark1.get(0).getY();
            float hand1z0=landmark1.get(0).getZ();

            float hand2x0=landmark2.get(0).getX();
            float hand2y0=landmark2.get(0).getY();
            float hand2z0=landmark2.get(0).getZ();
            float distance0_0=
                    (hand1x0-hand2x0)*(hand1x0-hand2x0)+
                            (hand1y0-hand2y0)*(hand1y0-hand2y0)+
                            (hand1z0-hand2z0)*(hand1z0-hand2z0);
            float hand1x17=landmark1.get(17).getX();
            float hand1y17=landmark1.get(17).getY();
            float hand1z17=landmark1.get(17).getZ();

            float distance0_17=
                    (hand1x0-hand1x17)*(hand1x0-hand1x17)+
                            (hand1y0-hand1y17)*(hand1y0-hand1y17)+
                            (hand1z0-hand1z17)*(hand1z0-hand1z17);
            float zero=0;
            if(distance0_17-distance0_0>zero){
                return "flower";
            }
        }
        return "none";
    }
    //显示
//    class MessageShow extends Thread{
//        @Override
//        public void run() {
//            Runnable runnable = new Runnable() {
//                @Override
//                public void run() {
//                    button_test_camera.setText(detect_res);
//                    if("Love".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.love128);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("Love2".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.love2);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("GOOD".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.good);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("OK".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.finish2);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("fist".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.fist);
//                        img_love.setVisibility(View.VISIBLE);
//                    } else if("数字1".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.one);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("数字2".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.ye128);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else if("flower".equals(detect_res)){
//                        img_love.setImageResource(R.drawable.flower);
//                        img_love.setVisibility(View.VISIBLE);
//                    }else{
//                        img_love.setVisibility(View.INVISIBLE);
//                    }
//                }
//            };
//            uiHandler.post(runnable);
//        }
//    }

//    class ShowPic extends Thread{
//        @Override
//        public void run() {
//            Runnable runnable = new Runnable() {
//                @Override
//                public void run() {
//                    img_love.setVisibility(View.INVISIBLE);
//                }
//            };
//            uiHandler.post(runnable);
//        }
//    }
}

