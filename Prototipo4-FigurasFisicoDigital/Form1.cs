using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.ComponentModel;
using System.Data;

using System.Linq;
using System.Text;



using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.OCR;
using Emgu.CV.Cvb;
using Emgu.CV.Cuda;
//using Emgu.CV.UI;
using Emgu.CV.VideoStab;

//using iText.IO.Font;
//using iText.Kernel.Font;
//using iText.Kernel.Colors;
//using iText.Kernel.Pdf;
//using iText.Kernel.Pdf.Canvas;
//using iText.Kernel.Pdf.Canvas.Parser;
//using iText.Kernel.Pdf.Canvas.Parser.Listener;
//using iText.Layout;
//using iText.Layout.Element;

using iTextSharp.text;
using iTextSharp.text.pdf;
using iTextSharp.text.pdf.parser;
using System.IO;
//using iText.Kernel.Geom;
using System.Drawing.Imaging;
//using iText.IO.Image;
//using Image = iText.Layout.Element.Image;
using Org.BouncyCastle.Asn1.Esf;

namespace Prototipo4_FigurasFisicoDigital
{
    public partial class Form1 : Form
    {
        Image<Bgr, Byte> imge;
        VideoCapture _capture;
        private Mat _frame;
        private const int Threshold = 1;
        private const int ErodeIterations = 1;
        private const int DilateIterations = 7;
        private static MCvScalar drawingColor = new Bgr(System.Drawing.Color.Blue).MCvScalar;
        List<VectorOfPoint> resultados = new List<VectorOfPoint>();
        List<System.Drawing.Point> centrosCir = new List<System.Drawing.Point>();
        //List<VectorOfPoint> resRec = new List<VectorOfPoint>();
        //List<VectorOfPoint> resCir = new List<VectorOfPoint>();
        //List<VectorOfPoint> resTri = new List<VectorOfPoint>();
        //List<VectorOfPoint> resEli = new List<VectorOfPoint>();
        // Bitmap bmp;
        // List<Bitmap> imagenes = new List<Bitmap>();

        List <Triangle2DF> triangleList = new List<Triangle2DF>();
        List<RotatedRect> boxList = new List<RotatedRect>(); //a box is a rotated rectangle
        List<System.Drawing.Rectangle> rect = new List<System.Drawing.Rectangle>();
        List<CircleF> circleList = new List<CircleF>();
        List<Ellipse> ellipseList = new List<Ellipse>();
        private async void ProcessFrame(object sender, EventArgs e)
        {
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                _capture.Retrieve(_frame, 0);
                Image<Bgr, byte> imagen_aux = _frame.ToImage<Bgr, byte>();
                imagen_aux = imagen_aux.Rotate(180, new Bgr(0, 0, 0));
                pictureBox1.Image = imagen_aux.Bitmap;
                //pictureBox1.Image = _frame.Bitmap;
                double fps = 15;
                await Task.Delay(1000 / Convert.ToInt32(fps));

            }
        }
        public Form1()
        {
            InitializeComponent();
            _capture = new VideoCapture(1);


            _capture.ImageGrabbed += ProcessFrame;
            _frame = new Mat();
            if (_capture != null)
            {
                try
                {
                    _capture.Start();
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            resultados.Clear();
            for (int i = 0; i < 1; i++)
            {
                if (_frame.IsEmpty)
            {
                return;
            }

            Mat finalFrame = new Mat();
            Mat aux = new Mat();
                Mat aux2 = new Mat();
                Mat aux3= new Mat();
                Image<Bgr, byte> img = _frame.ToImage<Bgr, byte>();
                
                img = img.Rotate(180, new Bgr(0, 0, 0));
                
                //Transformar a espacio de color HSV
                Image<Hsv, Byte> hsvimg = img.Convert<Hsv, Byte>();

            //extract the hue and value channels
            Image<Gray, Byte>[] channels = hsvimg.Split();  //separar en componentes
            Image<Gray, Byte> imghue = channels[0];            //hsv, channels[0] es hue.
            Image<Gray, Byte> imgval = channels[2];            //hsv, channels[2] es value.

            //Filtro color
            //140 en adelante 
            //funciona  160
            Image<Gray, byte> huefilter = imghue.InRange(new Gray(150), new Gray(255));
            //Filtro colores menos brillantes
            Image<Gray, byte> valfilter = imgval.InRange(new Gray(100), new Gray(255));
            //Filtro de saturación - quitar blancos 
            channels[1]._ThresholdBinary(new Gray(20), new Gray(255)); // Saturacion

            //Unir los filtros para obtener la imagen
            Image<Gray, byte> colordetimg = huefilter.And(valfilter).And(channels[1]);//aqui habia un Not()
            //pictureBox2.Image = colordetimg.Bitmap;
            //colordetimg._Erode(1);
            //2 y 4
            Mat SE2 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 2), new System.Drawing.Point(1, 1));
            CvInvoke.MorphologyEx(colordetimg, aux, MorphOp.Erode, SE2, new System.Drawing.Point(-1, -1), 2, BorderType.Default, new MCvScalar(255));
            Mat SE3 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 3), new System.Drawing.Point(1, 1));
            CvInvoke.MorphologyEx(aux, aux2, MorphOp.Dilate, SE3, new System.Drawing.Point(-1, -1), 3, BorderType.Replicate, new MCvScalar(255));

                Mat SE = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 3), new System.Drawing.Point(-1, -1));
            CvInvoke.MorphologyEx(aux2, aux3, Emgu.CV.CvEnum.MorphOp.Close, SE, new System.Drawing.Point(-1, -1), 5, Emgu.CV.CvEnum.BorderType.Reflect, new MCvScalar(255));
           CvInvoke.MorphologyEx(aux3, aux3, Emgu.CV.CvEnum.MorphOp.Open, SE, new System.Drawing.Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Reflect, new MCvScalar(255));

             pictureBox3.Image = aux3.Bitmap;
            
                _frame.CopyTo(finalFrame);

                DetectObject(aux3, finalFrame);
                MessageBox.Show("tamano lista figuras " + resultados.Count);
                //pictureBox2.Image = finalFrame.Bitmap;

            }

            /*for(int i=0;i<resRec.Count;i++)
            {
                System.Drawing.Point[] pts = resRec[i].ToArray();

                var momentsI = CvInvoke.Moments(resRec[i]);
                int xI = (int)(momentsI.M10 / momentsI.M00);
                int yI = (int)(momentsI.M01 / momentsI.M00);
                for(int j=0;j < resRec.Count; j++)
                {
                    var momentsJ = CvInvoke.Moments(resRec[i]);
                    int xJ = (int)(momentsJ.M10 / momentsJ.M00);
                    int yJ = (int)(momentsJ.M01 / momentsJ.M00);
                    
                }
                
            }
            */

            //double fps = 15;
            //await Task.Delay(1000 / Convert.ToInt32(fps));

        }



        //  
        //COMIENZAN FUNCIONES DE EDDIE
        //
        private void DetectObject(Mat detectionFrame, Mat displayFrame)
        {
            System.Drawing.Rectangle box = new System.Drawing.Rectangle();
            Image<Bgr, byte> temp = detectionFrame.ToImage<Bgr, byte>();
            
            temp = temp.Rotate(180, new Bgr(0, 0, 0));
            Image<Bgr, Byte> buffer_im = displayFrame.ToImage<Bgr, Byte>();
            float a = buffer_im.Width;
            float b = buffer_im.Height;
            MessageBox.Show("El tamano camara es  W: "+ a.ToString()+" y H:" +  b.ToString());
            
            boxList.Clear();
            rect.Clear();
            triangleList.Clear();
            circleList.Clear();
            ellipseList.Clear();

            //transforma imagen
            //UMat uimage = new UMat();
            // CvInvoke.CvtColor(displayFrame, uimage, ColorConversion.Bgr2Gray);
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
              ///  IOutputArray hirarchy = null;
               /// CvInvoke.FindContours(detectionFrame, contours, hirarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);
                ///CvInvoke.Polylines(detectionFrame, contours, true, new MCvScalar(255, 0, 0), 2, LineType.FourConnected);
                Image<Bgr, Byte> resultadoFinal = displayFrame.ToImage<Bgr, byte>();

                resultadoFinal = resultadoFinal.Rotate(180, new Bgr(0, 0, 0));
               
                //Circulos
                //double cannyThreshold = 180.0;
                //double circleAccumulatorThreshold = 120;
                //CircleF[] circles = CvInvoke.HoughCircles(detectionFrame, HoughType.Gradient, 2.0, 20.0, cannyThreshold, circleAccumulatorThreshold, 5);

                /// if (contours.Size > 0)
                ///{
                double maxArea = 1000;
                    int chosen = 0;
                    VectorOfPoint contour = null;
                    /*
                    for (int i = 0; i < contours.Size; i++)
                    {
                        contour = contours[i];

                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            System.Drawing.Rectangle rect = new System.Drawing.Rectangle();
                            //  maxArea = area;
                            chosen = i;
                            //}
                            //}

                            //Boxes
                            VectorOfPoint hullPoints = new VectorOfPoint();
                            VectorOfInt hullInt = new VectorOfInt();

                            CvInvoke.ConvexHull(contours[chosen], hullPoints, true);
                            CvInvoke.ConvexHull(contours[chosen], hullInt, false);

                            Mat defects = new Mat();

                            if (hullInt.Size > 3)
                                CvInvoke.ConvexityDefects(contours[chosen], hullInt, defects);

                            box = CvInvoke.BoundingRectangle(hullPoints);
                            CvInvoke.Rectangle(displayFrame, box, drawingColor);//Box rectangulo que encierra el area mas grande
                                                                                // cropbox = crop_color_frame(displayFrame, box);

                            buffer_im.ROI = box;

                            Image<Bgr, Byte> cropped_im = buffer_im.Copy();
                            //pictureBox8.Image = cropped_im.Bitmap;
                            System.Drawing.Point center = new System.Drawing.Point(box.X + box.Width / 2, box.Y + box.Height / 2);//centro  rectangulo MOUSE
                            System.Drawing.Point esquina_superiorI = new System.Drawing.Point(box.X, box.Y);
                            System.Drawing.Point esquina_superiorD = new System.Drawing.Point(box.Right, box.Y);
                            System.Drawing.Point esquina_inferiorI = new System.Drawing.Point(box.X, box.Y + box.Height);
                            System.Drawing.Point esquina_inferiorD = new System.Drawing.Point(box.Right, box.Y + box.Height);
                            CvInvoke.Circle(displayFrame, esquina_superiorI, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_superiorD, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_inferiorI, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_inferiorD, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, center, 5, new MCvScalar(0, 0, 255), 2);
                            VectorOfPoint start_points = new VectorOfPoint();
                            VectorOfPoint far_points = new VectorOfPoint();






                        }
                    }
                    */
                    //Dibuja borde rojo
                    var temp2 = temp.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinary(new Gray(20), new Gray(255));
                    temp2 = temp2.Rotate(180, new Gray(0));
                    VectorOfVectorOfPoint contorno = new VectorOfVectorOfPoint();
                    Mat mat = new Mat();
                    CvInvoke.FindContours(temp2, contorno, mat, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                

                for (int i = 0; i < contorno.Size; i++)
                    {

                        VectorOfPoint approxContour = new VectorOfPoint();
                        double perimetro = CvInvoke.ArcLength(contorno[i], true);
                        VectorOfPoint approx = new VectorOfPoint();
                        
                        VectorOfPointF approxF = new VectorOfPointF();
                        double area = CvInvoke.ContourArea(contorno[i]);
                        if (area > 5000)
                        {
                            CvInvoke.ApproxPolyDP(contorno[i], approx, 0.04 * perimetro, true);
                        // CvInvoke.DrawContours(displayFrame, contorno, i, new MCvScalar(255, 0, 0), 2);

                        //pictureBox4.Image = temp2.Bitmap;

                        var moments = CvInvoke.Moments(contorno[i]);
                            int x = (int)(moments.M10 / moments.M00);
                            int y = (int)(moments.M01 / moments.M00);



                            resultados.Add(approx);
                            bool isShape;
                            if (approx.Size == 3) //The contour has 3 vertices, it is a triangle
                            {
                                System.Drawing.Point[] pts = approx.ToArray();
                            double perimetro2 = CvInvoke.ArcLength(contorno[i], true);

                            double area2 = CvInvoke.ContourArea(contorno[i]);
                            double circularidad = 4 * Math.PI * area2 / Math.Pow(perimetro2, 2);
                             MessageBox.Show("circularidad triangulo" + circularidad);
                            MessageBox.Show("Es triangulo ");
                                    /*Triangle2DF triangle = new Triangle2DF(pts[0], pts[1], pts[2]);
                                    resultadoFinal.Draw(triangle, new Bgr(System.Drawing.Color.Cyan), 1);
                                    CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                    CvInvoke.PutText(resultadoFinal, "Triangle", new System.Drawing.Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                    resTri.Add(approx);*/
                               
                                    //MessageBox.Show("No es triangulo ");
                                    //Triangle2DF triangle = new Triangle2DF(pts[0], pts[1], pts[2]);
                                    //resultadoFinal.Draw(triangle, new Bgr(System.Drawing.Color.Red), 2);
                                    RotatedRect rectangle = CvInvoke.MinAreaRect(approx);
                                    CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                    resultadoFinal.Draw(rectangle, new Bgr(System.Drawing.Color.Cyan), 1);
                                    rect.Add(CvInvoke.BoundingRectangle(approx));



                        }
                            if (approx.Size == 4) //The contour has 4 vertices.
                            {
                            //RotatedRect tt = new RotatedRect(CvInvoke.MinAreaRect(approx).Center, CvInvoke.MinAreaRect(approx).Size, 270) ;
                            //boxList.Add(tt);

                            //Calcular si es cuadrado
                            System.Drawing.Rectangle rectAux = CvInvoke.BoundingRectangle(contorno[i]);
                            double ar = (double)rectAux.Width / rectAux.Height;

                            //Calcular circularidad
                            double perimetro2 = CvInvoke.ArcLength(contorno[i], true);
                            double area2 = CvInvoke.ContourArea(contorno[i]);
                            double circularidad = 4 * Math.PI * area2 / Math.Pow(perimetro2, 2);

                            MessageBox.Show("circularidad rect " + circularidad);
                            if (circularidad > 0.69 )
                            {
                                //Si la circularidad>0.6 y cumple proporcion es cuadrado
                                if (ar >= 0.8 && ar <= 1.0)
                                {
                                    MessageBox.Show("Cuadrado ");
                                    RotatedRect rectangle = CvInvoke.MinAreaRect(contorno[i]);
                                    CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                    resultadoFinal.Draw(rectangle, new Bgr(System.Drawing.Color.Cyan), 1);
                                    //CvInvoke.PutText(resultadoFinal, "Rectangle", new System.Drawing.Point(x, y),
                                    //Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                    rect.Add(CvInvoke.BoundingRectangle(approx));

                                }
                                //Es elipse 
                                else
                                {
                                    MessageBox.Show("parecia rectangulo pero era elipse ");
                                    Ellipse final_ellipse = new Ellipse(CvInvoke.MinAreaRect(contorno[i]).Center, CvInvoke.MinAreaRect(contorno[i]).Size, 0);
                                    Ellipse final_ellipseDibujo = new Ellipse(CvInvoke.MinAreaRect(contorno[i]).Center, CvInvoke.MinAreaRect(contorno[i]).Size, 90);
                                    ellipseList.Add(final_ellipse);

                                    //IConvexPolygonF poligono = CvInvoke.MinAreaRect(approx);
                                    //resultadoFinal.Draw(poligono, new Bgr(Color.Cyan), 1);
                                    resultadoFinal.Draw(final_ellipseDibujo, new Bgr(System.Drawing.Color.Cyan), 1);
                                    CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                    //CvInvoke.PutText(resultadoFinal, "Figura circular", new System.Drawing.Point(x, y),
                                      //      Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                }
                            }
                            //Es rectangulo
                            else
                            {
                                RotatedRect rectangle = CvInvoke.MinAreaRect(contorno[i]);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                resultadoFinal.Draw(rectangle, new Bgr(System.Drawing.Color.Cyan), 1);
                                //CvInvoke.PutText(resultadoFinal, "Rectangle", new System.Drawing.Point(x, y),
                                //Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                rect.Add(CvInvoke.BoundingRectangle(approx));


                            }


                            /* //prueba imagen de rectangulo
                            //--------------------------------------PART 1 : DRAWING STUFF IN A BITMAP------------------------------------------------------------------------------------
                            System.Drawing.Point[] pts = approx.ToArray();

                            System.Drawing.PointF[] mypoints = Array.ConvertAll(
                                 pts.ToArray<System.Drawing.Point>(),
                                 value => new System.Drawing.PointF(value.X, value.Y)
                               );

                            System.Drawing.Rectangle r = new System.Drawing.Rectangle(0, 0, CvInvoke.BoundingRectangle(approx).Width, CvInvoke.BoundingRectangle(approx).Height);
                            Pen blackPen = new Pen(System.Drawing.Color.FromArgb(255, 255, 0, 0), 1);
                            bmp = new Bitmap(r.Width+100,r.Height+10, PixelFormat.Format32bppArgb);
                            Graphics g = Graphics.FromImage(bmp);
                            g.DrawRectangle(blackPen, r); //rectangle 1
                            g.DrawPolygon(blackPen,mypoints);
                            System.Drawing.Rectangle rcrop = new System.Drawing.Rectangle(r.X, r.Y, r.Width + 10, r.Height + 10);//This is the cropping rectangle (bonding box adding 10 extra units width and height)

                            //Crop the model from the bmp
                            Bitmap src = bmp;
                           // Bitmap target = new Bitmap(r.Width, r.Height);
                            //using (Graphics gs = Graphics.FromImage(target))
                            //{
                              //  gs.DrawImage(src, rcrop, r, GraphicsUnit.Pixel);
                               // gs.Dispose();
                            //}
                            //--------------------------------------PART 2 : SAVING THE BMP AS JPG------------------------------------------------------------------------------------
                            src.Save("testOJO.jpg");*/







                        }
                            /* ELIMINAR 
                             * if (approx.Size == 5 )
                            {
                                System.Drawing.Point[] pts = approx.ToArray();
                               
                                //MessageBox.Show("Cantidad puntos poligono "+pts.Length);
                                //IConvexPolygonF poligono = CvInvoke.MinAreaRect(approx);
                                //resultadoFinal.Draw(poligono, new Bgr(Color.Cyan), 1);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 0), 1, LineType.AntiAlias);
                                CvInvoke.PutText(resultadoFinal, "Pentagon", new System.Drawing.Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                            }*/
                            if (approx.Size>=5)
                            {

                                double perimetro2 = CvInvoke.ArcLength(contorno[i], true);
                                double area2 = CvInvoke.ContourArea(contorno[i]);
                                double circularidad = 4 * Math.PI * area2 / Math.Pow(perimetro2,2);
                                MessageBox.Show("circularidad elipse " + circularidad);

                                Ellipse final_ellipse = new Ellipse(CvInvoke.MinAreaRect(contorno[i]).Center, CvInvoke.MinAreaRect(contorno[i]).Size,0);
                                Ellipse final_ellipseDibujo = new Ellipse(CvInvoke.MinAreaRect(contorno[i]).Center, CvInvoke.MinAreaRect(contorno[i]).Size, 90);
                                ellipseList.Add(final_ellipse);

                                //IConvexPolygonF poligono = CvInvoke.MinAreaRect(approx);
                                //resultadoFinal.Draw(poligono, new Bgr(Color.Cyan), 1);
                                resultadoFinal.Draw(final_ellipseDibujo, new Bgr(System.Drawing.Color.Cyan), 1);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                //CvInvoke.PutText(resultadoFinal, "Figura circular", new System.Drawing.Point(x, y),
                                  //      Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                               
                                   

                                
                            }


                            /* _Eliminar
                             * if (approx.Size > 6)
                            {

                                    double circularidad = 4 * Math.PI * area / (Math.Pow(2, perimetro));
                                MessageBox.Show("circularidad circulo "+circularidad);
                                    CvInvoke.PutText(resultadoFinal, "Circle", new System.Drawing.Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                    CircleF circle = CvInvoke.MinEnclosingCircle(approx);
                                    circleList.Add(circle);
                                    CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                    resultadoFinal.Draw(circle, new Bgr(System.Drawing.Color.Cyan), 1);
                               


                            }*/
                        }

                    }

                    pictureBox2.Image = resultadoFinal.Bitmap;
                    button2.Enabled = true;

                ///}

            
            }

        }

        public bool IsRealShape(VectorOfPoint approx, VectorOfPoint contorno)
        {

            //return true;
                double ratio = CvInvoke.MatchShapes(approx, contorno, ContoursMatchType.I3);
               // MessageBox.Show(" El ratio es  " + ratio + " bueno si es menor a 0.1");
                if (ratio < 0.1)
                {
                    //MessageBox.Show("Shape correcta con tamano "+approx.Size);
                    return true;
                }
                else
                {
                    return false;
                 

                }
            

            return false;
        }
        private void button2_Click(object sender, EventArgs e)
        {
            
            
            string oldFile = "opcion2X.pdf";
            string newFile = "temporal.pdf";
            
           // string oldFile = textBox6.Text;
            //string newFile = "Code2.pdf";

           /* PdfDocument pdfDoc = new PdfDocument(new PdfReader(oldFile), new PdfWriter(newFile));
            PdfCanvas canvas = new PdfCanvas(pdfDoc.GetFirstPage());
            iText.Kernel.Geom.Rectangle mediabox = pdfDoc.GetPage(1).GetMediaBox();
            double anchoPDF = mediabox.GetWidth();
            double altoPDF = mediabox.GetHeight();*/
            //MessageBox.Show("Medidas PDF ancho: "+ anchoPDF + " y alto: "+ altoPDF);
            //Sincronizar rectangulos
            //float incremento = 130;
            //Bitmap bmp = new Bitmap(1000, 1000, PixelFormat.Format32bppArgb);
           /* MemoryStream ms = new MemoryStream();
            bmp.Save(ms, ImageFormat.Png);
            byte[] bmpBytes = ms.ToArray();
            ImageData data = ImageDataFactory.Create( bmpBytes);
            
           
            canvas.AddImage(data,100,500,false);*/
           

            /*
            foreach (var item in rect)
            {

                
                float porcentaje = (item.Y * 100 / 640);
                double nuevoPtoY = (porcentaje * (841)) / 100;
                double proporcion = nuevoPtoY / item.Y;
                if (porcentaje < 50)
                {
                    incremento = 0;
                }
                else if (porcentaje > 50 && porcentaje < 80)
                {
                    incremento = 70;
                }
                else
                {
                    incremento = 100;
                }

               
                    //El x estaba con -50
                canvas.SetStrokeColor(new DeviceRgb(0, 0, 255))
                        .SetLineWidth(2)
                        .Rectangle(item.X-80, (841*0.9)-(nuevoPtoY*proporcion), item.Width*1.3, item.Height*proporcion)
                        .Stroke();
            }
            */
            //Sincronizar triangulos
            /*foreach (var triangle in triangleList)
            {
                canvas.SetStrokeColor(new DeviceRgb(0, 0, 255));
                PointF[] vertices = triangle.GetVertices();
                MessageBox.Show("estos son los vertices 0: " + vertices[0] + " 1: " +vertices[1] + " 2:" + vertices[2]);
                double area = triangle.Area;
                double h = (2 * area / (vertices[0].X - vertices[2].X));
                MessageBox.Show("esto es el area y esto el h "+area+ " " + h);
                double x = (vertices[2].X+(vertices[0].X - vertices[2].X));
                double y = vertices[2].Y;
                canvas.MoveTo(x, y);
                canvas.LineTo(vertices[1].X, (vertices[1].Y+(h*2)));
                canvas.LineTo((vertices[0].X-(vertices[0].X - vertices[2].X)), vertices[0].Y);
                canvas.Stroke();
            }*/

            //Sincronizar circulos
          /*  foreach (var circle in circleList)
            {
                double x = circle.Center.X;
                double y = circle.Center.Y;
                double r = circle.Radius;
                double porcentaje = (y * 100 / 640);
                double nuevoPtoY = (porcentaje * (841*0.9)) / 100;
                double proporcion = nuevoPtoY / y;
                // Setting color to the circle
                canvas.SetStrokeColor(new DeviceRgb(0, 255, 0));
                // creating a circle
                canvas.Circle(x-50, (841*0.9)-(nuevoPtoY), r*proporcion);

                // Filling the circ
                //canvas.Fill();
                canvas.Stroke();
            }*/


            //Sincronizar elipses y circulos
           /* foreach (var ellipse in ellipseList)
            {
                System.Drawing.Rectangle rr = ellipse.RotatedRect.MinAreaRect();
                double porcentajeB = (rr.Bottom * 100 / 640);
                double nuevoPtoB = (porcentajeB * (841)) / 100;
                double proporcionB = nuevoPtoB / rr.Bottom;

                double porcentajeT = (rr.Top * 100 / 640);
                double nuevoPtoT = (porcentajeT * (841)) / 100;
                double proporcionT = nuevoPtoB / rr.Top;

                canvas.SetStrokeColor(new DeviceRgb(255, 0, 255))
                    
                    .Ellipse( rr.Left-80, (841*0.9)-(nuevoPtoB), (rr.Right*1.1)-80, (841*0.9)-(nuevoPtoT))
                    .Stroke();
            }


                pdfDoc.Close();
            */



            PdfReader reader = new PdfReader(oldFile);
            var pageSize = reader.GetPageSize(1);
            Console.WriteLine("Tamano de pagina PDF "+pageSize);
            PdfStamper stamper = new PdfStamper(reader, new FileStream(newFile, FileMode.Append));
            
            
            PdfContentByte contentunder = stamper.GetUnderContent(1);
            
            //int rot;
            //rot = reader.GetPageRotation(1);
            //PdfDictionary pageDict;
            //pageDict = reader.GetPageN(1);
            //pageDict.Put(PdfName.ROTATE, new PdfNumber(rot + 90));
            
            
            float incremento = 130;
            foreach (var item in rect)
            {
                //float center1 = item.X+(item.X+item.Width)/2;
                //float center2 = item.Y + (item.Y + item.Height) / 2;
                //RotatedRect aux = new RotatedRect(new PointF(item.X, item.Y), item.Size, 180);
                //PointF[] puntos = aux.GetVertices();
                //float puntoX = puntos[0].X;
                //float puntoY = puntos[0].Y;
                //float tamanoH = item.Size.Height;
                //float tamanoW = item.Size.Width;
               

                float porcentaje = (item.Y * 100 / 640);
                double nuevoPtoY = (porcentaje * (841)) / 100;
                double proporcion = nuevoPtoY / item.Y;
                if (porcentaje < 50)
                {
                    incremento = 0;
                }
                else if (porcentaje > 50 && porcentaje < 80)
                {
                    incremento = 70;
                }
                else
                {
                    incremento = 100;
                }

                MessageBox.Show("Info rectangulo real : " + item.X + ", puntoY : " + item.Y + ", tamanoH : " + item.Height + ", tamanoW : " + item.Width);

                contentunder.SetColorStroke(BaseColor.YELLOW);
                ////Antigua coordenada contentunder.Rectangle(item.X - 50, (841 * 0.85) - (nuevoPtoY * proporcion)+incremento, item.Width * 1.3, item.Height * proporcion);
                ////nueva coordenada
                double propX = ((item.X * 100) / 640);
                double ptoX2 = ((propX * 792) / 100);
                double propY = ((item.Y * 100) / 480);
                double ptoY2 = ((propY * 612) / 100);
                double propArea = 1.58;
                double propW = ((item.Width) * 100) / 640;
                double ptoW2 = ((propW * 792) / 100);
                double propH = ((item.Height * 100) / 480);
                double ptoH2 = ((propH * 612) / 100);
                contentunder.Rectangle(ptoX2+30, (612-ptoY2-(ptoH2-30)), ptoW2-30, ptoH2-30);

                //contentunder.Rectangle(puntoX, puntoY, tamanoW, tamanoH);
                contentunder.Stroke();
                
            }

            //Sincronizar circulos
              foreach (var circle in circleList)
              {
                /*double x = circle.Center.X;
                double y = circle.Center.Y;
                double r = circle.Radius;
                double porcentaje = (y * 100 / 640);
                double nuevoPtoY = (porcentaje * (841*0.9)) / 100;
                double proporcion = nuevoPtoY / y;

              // Setting color to the circle
              contentunder.SetColorStroke(BaseColor.MAGENTA);
              // creating a circle
              contentunder.Circle(x, (841*0.9)-(nuevoPtoY), r*proporcion);*/
                double propX = ((circle.Center.X * 100) / 640);
                double ptoX2 = ((propX * 792) / 100);
                double propY = ((circle.Center.Y * 100) / 480);
                double ptoY2 = ((propY * 612) / 100);
                double r = circle.Radius;
                // Setting color to the circle
                contentunder.SetColorStroke(BaseColor.MAGENTA);
                // creating a circle
                contentunder.Circle(ptoX2, 612-ptoY2-r, r * 1.58);
                // Filling the circ
                //canvas.Fill();
                contentunder.Stroke();
              }


                    //Sincronizar elipses y circulos
             foreach (var ellipse in ellipseList)
             {
                 System.Drawing.Rectangle rr = ellipse.RotatedRect.MinAreaRect();
                double propLeft = ((rr.Left * 100) / 640);
                double ptoLeft2 = ((propLeft * 792) / 100);

                double propRight = ((rr.Right * 100) / 640);
                double ptoRight2 = ((propRight * 792) / 100);

                double propBottom = ((rr.Bottom* 100) / 480);
                double ptoBottom2 = ((propBottom * 612) / 100);

                double propTop = ((rr.Top * 100) / 480);
                double ptoTop2 = ((propTop * 612) / 100);

                contentunder.SetColorStroke(BaseColor.GREEN);
                contentunder.Ellipse(ptoLeft2, 612-ptoBottom2, ptoRight2, 612-ptoTop2);
                /*double porcentajeB = (rr.Bottom * 100 / 640);
                double nuevoPtoB = (porcentajeB * (841)) / 100;
                double proporcionB = nuevoPtoB / rr.Bottom;

                double porcentajeT = (rr.Top * 100 / 640);
                double nuevoPtoT = (porcentajeT * (841)) / 100;
                double proporcionT = nuevoPtoB / rr.Top;

               contentunder.SetColorStroke(BaseColor.GREEN);
               contentunder.Ellipse(rr.Left - 50, (841 * 0.9) - (nuevoPtoB), (rr.Right * 1.1) - 80, (841 * 0.9) - (nuevoPtoT));*/
                contentunder.Stroke();
             }

 stamper.Close();
 reader.Close();

 File.Replace(@newFile, oldFile, @"backup.pdf.bac");

 //File.Replace(@"temporal.pdf", @"ejemploOK.pdf", @"backup.pdf.bac");
 MessageBox.Show("Pdf modificado con exito, se ha guardado un backup de la versión anterior ");
 axAcroPDF1.src = "C:\\Users\\Denisse\\Desktop\\prototipos\\Prototipo4-FigurasFisicoDigital\\Prototipo4-FigurasFisicoDigital\\bin\\Debug\\opcion2X.pdf";

}

private void button3_Click(object sender, EventArgs e)
{

 string oldFile = "opcion2X.pdf";
 string backFile = "opcion2X - copia.pdf";

 File.Delete(oldFile);
 File.Copy(backFile, oldFile, true);
 MessageBox.Show("PDF restaurado");

}
}
}
