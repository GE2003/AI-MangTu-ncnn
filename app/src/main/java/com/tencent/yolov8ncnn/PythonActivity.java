package com.tencent.yolov8ncnn;

import android.app.Activity;
import android.os.Bundle;


import android.view.SurfaceView;
import androidx.annotation.Nullable;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class PythonActivity extends Activity {


    private SurfaceView surfaceView;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_python);

        // 找到布局文件中定义的视图组件
        surfaceView = (SurfaceView) this.findViewById(R.id.surfaceView);
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(PythonActivity.this));
        }

        Python python = Python.getInstance();
        PyObject pyObject = python.getModule("main");
        pyObject.callAttr("detect");

    }
}
