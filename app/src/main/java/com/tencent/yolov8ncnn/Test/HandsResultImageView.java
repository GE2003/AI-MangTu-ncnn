package com.tencent.yolov8ncnn.Test;


import android.content.Context;
import android.graphics.*;
import androidx.appcompat.widget.AppCompatImageView;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsResult;

import java.util.ArrayList;
import java.util.List;
//这里的代码是用来处理单个图片的结果
/** An ImageView implementation for displaying {@link HandsResult}. */
public class HandsResultImageView extends AppCompatImageView {
    private static final String TAG = "HandsResultImageView";

    private static final int LEFT_HAND_CONNECTION_COLOR = Color.parseColor("#30FF30");
    private static final int RIGHT_HAND_CONNECTION_COLOR = Color.parseColor("#FF3030");
    private static final int CONNECTION_THICKNESS = 8; // Pixels
    private static final int LEFT_HAND_HOLLOW_CIRCLE_COLOR = Color.parseColor("#30FF30");
    private static final int RIGHT_HAND_HOLLOW_CIRCLE_COLOR = Color.parseColor("#FF3030");
    private static final int HOLLOW_CIRCLE_WIDTH = 5; // Pixels
    private static final int LEFT_HAND_LANDMARK_COLOR = Color.parseColor("#FF3030");
    private static final int RIGHT_HAND_LANDMARK_COLOR = Color.parseColor("#30FF30");
    private static final int LANDMARK_RADIUS = 10; // Pixels
    private Bitmap latest;

    //    传入一个页面 MainActivity的页面是activity_main,是context
    public HandsResultImageView(Context context) {
        super(context);
        setScaleType(ScaleType.FIT_CENTER);
    }

    /**
     * Sets a {@link HandsResult} to render.
     *
     * @param result a {@link HandsResult} object that contains the solution outputs and the input
     *     {@link Bitmap}.
     */
    public void setHandsResult(HandsResult result) {
//        这里接收手势检测的结果
        if (result == null) {
            return;
        }
        Bitmap bmInput = result.inputBitmap();
        int width = bmInput.getWidth();
        int height = bmInput.getHeight();
        latest = Bitmap.createBitmap(width, height, bmInput.getConfig());
        Canvas canvas = new Canvas(latest);

        canvas.drawBitmap(bmInput, new Matrix(), null);
//        检测到手的数量
        int numHands = result.multiHandLandmarks().size();

        for (int i = 0; i < numHands; ++i) {
            drawLandmarksOnCanvas(
                    result.multiHandLandmarks().get(i).getLandmarkList(),
                    result.multiHandedness().get(i).getLabel().equals("Left"),
                    canvas,
                    width,
                    height);
            drawClassificationOnCanvas(result.multiHandLandmarks().get(i).getLandmarkList(),canvas);
        }
    }


    /** Updates the image view with the latest {@link HandsResult}. */
    public void update() {
        postInvalidate();
        if (latest != null) {
            setImageBitmap(latest);
        }
    }

    private void drawClassificationOnCanvas(List<NormalizedLandmark> handLandmarkList,Canvas canvas) {
        String resstr = handDetection(handLandmarkList);
        Paint classificationPaint = new Paint();
        classificationPaint.setColor(Color.parseColor("#c75450"));
        classificationPaint.setStrokeWidth(12);
        classificationPaint.setTextSize(100);
        classificationPaint.setStyle(Paint.Style.FILL);
        System.out.println("----------printText");
        canvas.drawText(resstr,10,100,classificationPaint);
    }

    private void drawLandmarksOnCanvas(
            List<NormalizedLandmark> handLandmarkList,
            boolean isLeftHand,
            Canvas canvas,
            int width,
            int height) {
        // Draw connections.
//        画线
        for (Hands.Connection c : Hands.HAND_CONNECTIONS) {
            Paint connectionPaint = new Paint();
            connectionPaint.setColor(
                    isLeftHand ? LEFT_HAND_CONNECTION_COLOR : RIGHT_HAND_CONNECTION_COLOR);
            connectionPaint.setStrokeWidth(CONNECTION_THICKNESS);
            NormalizedLandmark start = handLandmarkList.get(c.start());
            NormalizedLandmark end = handLandmarkList.get(c.end());
            canvas.drawLine(
                    start.getX() * width,
                    start.getY() * height,
                    end.getX() * width,
                    end.getY() * height,
                    connectionPaint);
        }
//        画点 红点
        Paint landmarkPaint = new Paint();
        landmarkPaint.setColor(isLeftHand ? LEFT_HAND_LANDMARK_COLOR : RIGHT_HAND_LANDMARK_COLOR);
        // Draws landmarks.
        for (NormalizedLandmark landmark : handLandmarkList) {
            canvas.drawCircle(
                    landmark.getX() * width, landmark.getY() * height, LANDMARK_RADIUS, landmarkPaint);
        }
        // Draws hollow circles around landmarks.
//        画圈，红点上的圈圈
        landmarkPaint.setColor(
                isLeftHand ? LEFT_HAND_HOLLOW_CIRCLE_COLOR : RIGHT_HAND_HOLLOW_CIRCLE_COLOR);
        landmarkPaint.setStrokeWidth(HOLLOW_CIRCLE_WIDTH);
        landmarkPaint.setStyle(Paint.Style.STROKE);
        for (NormalizedLandmark landmark : handLandmarkList) {
            canvas.drawCircle(
                    landmark.getX() * width,
                    landmark.getY() * height,
                    LANDMARK_RADIUS + HOLLOW_CIRCLE_WIDTH,
                    landmarkPaint);
        }
    }

    //    手势检测
    private String handDetection(List<NormalizedLandmark> handLandmarkList){

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
                code
                        +="1";
            }else{
                code=code+"0";
            }
        }
        String res="";
        switch (code){
            case "11111":
                res="数字0/拳头";
                break;
            case "10111":
                res="数字1";
                break;
            case "10011":
                res="数字2/剪刀";
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
            default:
                res="none";
                break;
        }

        return res;
    }
}
