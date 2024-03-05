// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.*;
import android.widget.*;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutioncore.VideoInput;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;
import com.tencent.yolov8ncnn.Test.HandsResultGlRenderer;
import com.tencent.yolov8ncnn.Test.HandsResultImageView;
import com.tencent.yolov8ncnn.Test.TestActivity;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback
{

    public String detect_res="";
    private Handler uiHandler = new Handler();
    private static final String TAG = "MainActivity";

    private Hands hands;
    // Run the pipeline and the model inference on GPU or CPU.
    private static final boolean RUN_ON_GPU = true;
    private ImageView bmshow;

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












  ///////////////////////////////////////////////////////

    public static final int REQUEST_CAMERA = 100;

    private Yolov8Ncnn yolov8ncnn = new Yolov8Ncnn();
    private int facing = 1;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;
  private Bitmap bmp = null;
    private SurfaceView cameraView;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView =  findViewById(R.id.cameraview);

       cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
       cameraView.getHolder().addCallback(this);
       cameraView.setDrawingCacheEnabled(true);
       cameraView.buildDrawingCache(true);


//        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
//        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View arg0) {
//
//                int new_facing = 1 - facing;
//
//                yolov8ncnn.closeCamera();
//
//                yolov8ncnn.openCamera(new_facing);
//
//                facing = new_facing;
//            }
//        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();

      //  setupStaticImageDemoUiComponents();

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
//   实时渲染显示组件
    private void setupLiveDemoUiComponents() {
//        Button startCameraButton = findViewById(R.id.start_camera);
//        startCameraButton.setOnClickListener(
//                v -> {
//                    if (inputSource == InputSource.CAMERA) {
//                        return;
//                    }
//                    stopCurrentPipeline();
//                    setupStreamingModePipeline(InputSource.CAMERA);
//                });
        if (inputSource == InputSource.CAMERA) {
            return;
        }
        stopCurrentPipeline();
        setupStreamingModePipeline(InputSource.CAMERA);
    }
    private void startCamera() {
        cameraInput.start(
                this,
                hands.getGlContext(),
                CameraInput.CameraFacing.FRONT,
                glSurfaceView.getWidth(),
                glSurfaceView.getHeight());
    }
    /** Sets up core workflow for streaming mode. */
    private void setupStreamingModePipeline(InputSource inputSource) {
//        Bitmap mScreenBitmap = null;
//
//            //需要截取的长和宽
//            int outWidth = cameraView.getWidth();
//            int outHeight = cameraView.getHeight();
//
//            mScreenBitmap = Bitmap.createBitmap(30, 30,Bitmap.Config.ARGB_8888);
//            PixelCopy.request(cameraView, mScreenBitmap, new PixelCopy.OnPixelCopyFinishedListener() {
//                @Override
//                public void onPixelCopyFinished(int copyResult){
//                    if (PixelCopy.SUCCESS == copyResult) {
//
//                        Log.i("gyx","SUCCESS ");
//                    } else {
//                        Log.i("gyx","FAILED");
//                        // onErrorCallback()
//                    }
//                }
//            }, new Handler());






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
            System.out.println("视频帧————————"+bmp);
            cameraInput.setNewFrameListener(textureFrame -> hands.send(bmp));
        } else if (inputSource == InputSource.VIDEO) {
            videoInput = new VideoInput(this);
            videoInput.setNewFrameListener(textureFrame -> hands.send(bmp));
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
        FrameLayout frameLayout = findViewById(R.id.preview_main);
        //   imageView.setVisibility(View.GONE);
        frameLayout.removeAllViewsInLayout();
        frameLayout.addView(glSurfaceView);

        glSurfaceView.setVisibility(View.VISIBLE);
        frameLayout.requestLayout();
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
        LandmarkProto.NormalizedLandmark wristLandmark =
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
        LandmarkProto.Landmark wristWorldLandmark =
                result.multiHandWorldLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
        Log.i(
                TAG,
                String.format(
                        "MediaPipe Hand wrist world coordinates (in meters with the origin at the hand's"
                                + " approximate geometric center): x=%f m, y=%f m, z=%f m",
                        wristWorldLandmark.getX(), wristWorldLandmark.getY(), wristWorldLandmark.getZ()));
    }
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
    private String baseHandCount(List<LandmarkProto.NormalizedLandmark> handLandmarkList){

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
    private String twoHandDetect(List<LandmarkProto.NormalizedLandmark> landmark1, List<LandmarkProto.NormalizedLandmark> landmark2){
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



    public static int getBitmapSize(Bitmap bitmap) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {    //API 19
            return bitmap.getAllocationByteCount();
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB_MR1) {//API 12
            return bitmap.getByteCount();
        }
        // 在低版本中用一行的字节x高度
        return bitmap.getRowBytes() * bitmap.getHeight();                //earlier version
    }











    private void reload()
    {
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        boolean ret_init2 = yolov8ncnn.loadModel2(getAssets(), current_model, current_cpugpu);


        if (!ret_init)
        {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }

        if (!ret_init2)
        {
            Log.e("MainActivity", "yolov8ncnn loadModel2 failed");
        }
    }
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        //设置视频流输出窗口
        yolov8ncnn.setOutputWindow(holder.getSurface());


            bmp = Bitmap.createBitmap(cameraView.getDrawingCache());
        System.out.println("执行");

        int bitmapSize = getBitmapSize(bmp);
        System.out.println("大小"+bmp.getByteCount());
        setupLiveDemoUiComponents();
        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        int screenWidth=dm.widthPixels;
        if(bmp.getWidth()<=screenWidth){

        }else{
             bmp=Bitmap.createScaledBitmap(bmp, screenWidth, bmp.getHeight()*screenWidth/bmp.getWidth(), true);

        }
    }
    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }
    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }
    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
        yolov8ncnn.openCamera(facing);

    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolov8ncnn.closeCamera();
    }
}
