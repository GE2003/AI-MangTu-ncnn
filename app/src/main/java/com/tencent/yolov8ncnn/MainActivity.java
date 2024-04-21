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
import android.icu.text.UFormat;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
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
import java.util.*;


public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback
{

    public String detect_res="";
    private Handler uiHandler = new Handler();
    private static final String TAG = "MainActivity";

    private Hands hands;
    // Run the pipeline and the model inference on GPU or CPU.
    private static final boolean RUN_ON_GPU = true;
    private ImageView bmshow;
    private TTSUtils tts;

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
    List<String> list = new ArrayList<>();
  private Bitmap bmp = null;
    private SurfaceView cameraView;
  private Timer mTimer;
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
        mTimer = new Timer();
        //5s响一次
        mTimer.schedule(timerTask, 0, 1000);
     //  yolov8ncnn.speakText("盲图AI测试");

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
    TimerTask timerTask = new TimerTask() {

        @Override
        public void run() {
            Message msg = new Message();
            msg.what = 1;
            handler.sendMessage(msg);
        }
    };

    Handler handler = new Handler(){
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case 1:
                    //需要循环执行的代码
                    setTextSpeak();
                    break;
                default:
                    break;
            }
        };
    };

    private void setTextSpeak() {
        String label = yolov8ncnn.checkPointIfInDetectBox();
        tts = TTSUtils.getInstance(this);
        System.out.println(label);

            if (label.equals("WNke")){

            tts.playText("蜗牛的壳");
                list.clear();

            }else if (label.equals("WNtoubu")){

        tts.playText("蜗牛的头部");
                list.clear();
            }else if (label.equals("WNshenti")){
        tts.playText("蜗牛的身体");
                list.clear();
            }else {
                list.add(label);
                if (list.size()==7){

        tts.playText(label);
        list.clear();
                }
            }



    }


    private void reload()
    {
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        boolean ret_init2 = yolov8ncnn.loadModel2(getAssets(), current_model, 0);


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
        setTextSpeak();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        tts.stopSpeak();
        yolov8ncnn.closeCamera();
    }
}
