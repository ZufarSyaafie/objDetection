import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class TemplateMatching {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String imagePath = "src/test/Test.jpg";

        // Membaca gambar sumber
        Mat sourceImage = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        if (sourceImage.empty()) {
            System.out.println("Error: Cannot load source image.");
            return;
        }

        // Mendefinisikan template dan label yang sesuai
        String[][] templates = {
                {"src/template/apel.jpg", "Apel"},
                {"src/template/banana.jpg", "Pisang"},
                {"src/template/banana2.jpg", "Pisang"}
        };

        // Create a copy of the original image for display and convert to BGR
        Mat displayImage = new Mat();
        Imgproc.cvtColor(sourceImage, displayImage, Imgproc.COLOR_GRAY2BGR);

        List<Rect> boundingBoxes = new ArrayList<>();

        for (String[] templateData : templates) {
            Mat templateImage = Imgcodecs.imread(templateData[0], Imgcodecs.IMREAD_GRAYSCALE);
            if (templateImage.empty()) {
                System.out.println("Error: Cannot load template image " + templateData[0]);
                continue;
            }

            // Mengubah ukuran template agar lebih kecil jika perlu
            if (templateImage.cols() > sourceImage.cols() || templateImage.rows() > sourceImage.rows()) {
                Mat resizedTemplate = new Mat();
                Size newSize = new Size(sourceImage.cols() * 0.5, sourceImage.rows() * 0.5);
                Imgproc.resize(templateImage, resizedTemplate, newSize);
                templateImage = resizedTemplate; // Ganti templateImage dengan resizedTemplate
            }

            String label = templateData[1];
            Mat outputImage = new Mat();

            // Melakukan template matching
            Imgproc.matchTemplate(sourceImage, templateImage, outputImage, Imgproc.TM_CCOEFF_NORMED);

            // Menentukan threshold
            double threshold = 0.6;

            // List untuk menyimpan bounding box yang ditemukan dari template saat ini
            List<Rect> currentBoxes = new ArrayList<>();

            // Melakukan deteksi semua titik yang sesuai dengan threshold
            for (int y = 0; y < outputImage.rows(); y++) {
                for (int x = 0; x < outputImage.cols(); x++) {
                    // Cek apakah nilai kecocokan pada titik ini lebih besar dari threshold
                    if (outputImage.get(y, x)[0] >= threshold) {
                        Point matchLoc = new Point(x, y);
                        Rect boundingBox = new Rect(
                                new Point(matchLoc.x, matchLoc.y),
                                new Point(matchLoc.x + templateImage.cols(), matchLoc.y + templateImage.rows())
                        );
                        currentBoxes.add(boundingBox);
                        // Menampilkan hasil deteksi di konsol
                        System.out.println("Terdeksi: " + label + " di (" + matchLoc.x + ", " + matchLoc.y + ")");
                    }
                }
            }

            // Lakukan Non-Maximum Suppression (NMS) untuk menggabungkan bounding box yang tumpang tindih
            List<Rect> finalBoxes = nonMaximumSuppression(currentBoxes);  // Threshold IoU = 0.5
            boundingBoxes.addAll(finalBoxes); // Menambahkan semua kotak akhir ke daftar boundingBoxes

            // Menandai hasil akhir pada gambar
            for (Rect box : finalBoxes) {
                Imgproc.rectangle(displayImage, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
                Imgproc.putText(displayImage, label, new Point(box.x, box.y + 20),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(255, 0, 0), 2);
            }
        }

        // Menampilkan Jumlah Objek yang terdeteksi
        System.out.println("\nJumlah Objek yang terdeteksi: " + boundingBoxes.size());


        // Memperbesar gambar keluaran sebelum menampilkannya
        Mat enlargedImage = new Mat();
        Size newSize = new Size(displayImage.cols() * 2, displayImage.rows() * 2); // Menggandakan ukuran
        //Tampilkan di GUI Jumlah Objek yang terdeteksi
        Imgproc.putText(displayImage, "Objek: " + boundingBoxes.size(), new Point(0, 90),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, new Scalar(255, 0, 0), 1);
        Imgproc.resize(displayImage, enlargedImage, newSize);


        // Menampilkan hasil sekali saja setelah semua template diproses
        HighGui.imshow("Matched Result", enlargedImage);
        HighGui.waitKey();
        HighGui.destroyAllWindows();
    }

    // Fungsi untuk melakukan Non-Maximum Suppression (NMS)
    private static List<Rect> nonMaximumSuppression(List<Rect> boxes) {
        List<Rect> finalBoxes = new ArrayList<>();

        // Selama masih ada box dalam list, terus lakukan NMS
        while (!boxes.isEmpty()) {
            // Ambil box pertama
            Rect bestBox = boxes.removeFirst();
            finalBoxes.add(bestBox);

            // Cek overlap untuk setiap box lainnya
            // Jika IoU lebih besar dari threshold, hapus box yang tumpang tindih
            boxes.removeIf(box -> computeIoU(bestBox, box) > 0.5);
        }

        return finalBoxes;
    }

    // Fungsi untuk menghitung Intersection over Union (IoU)
    private static double computeIoU(Rect box1, Rect box2) {
        // Hitung koordinat titik overlap
        double x1 = Math.max(box1.x, box2.x);
        double y1 = Math.max(box1.y, box2.y);
        double x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        double y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

        // Area overlap
        double overlapArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

        // Total area dari kedua bounding box
        double box1Area = box1.width * box1.height;
        double box2Area = box2.width * box2.height;

        // Hitung IoU
        return overlapArea / (box1Area + box2Area - overlapArea);
    }
}