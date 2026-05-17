# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QFileDialog)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QPixmap

LABEL_NAMES = ['F-16', 'F-35', 'TB2', 'MiG-29', 'Su-57', 'Heron', 'Civil']
IFF_MAP = {
    0: ('DOST',  '#3FB950', (63, 185, 80)),
    1: ('DOST',  '#3FB950', (63, 185, 80)),
    2: ('DOST',  '#3FB950', (63, 185, 80)),
    3: ('DÜŞMAN','#F85149', (248, 81, 73)),
    4: ('DÜŞMAN','#F85149', (248, 81, 73)),
    5: ('DÜŞMAN','#F85149', (248, 81, 73)),
    6: ('SİVİL', "#1F4FD3", (31, 79, 211)), 
}


IFF_COLORS = {
    'F-16':  "#37E04E", 'F-35': "#37E04E", 'TB2':   "#37E04E",
    'MiG-29':"#DC1D1D", 'Su-57':"#DC1D1D", 'Heron': "#DC1D1D",
    'Civil': "#1F4FD3"
}
IMG_SIZE = 224
BAR_MAX_WIDTH = 180


class IFFApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.model = None
        
        # ÇERÇEVESİZ VE ŞEFFAF DÜZEN (MacOS gri astarını ezer geçer)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        self._drag_active = False
        self._drag_start_pos = QPoint()
        
        self.init_ui()

    def get_dominant_color(self, image_path):
        """Gerçek Spotify hissi için uçağın en canlı rengini yakalar."""
        try:
            img = Image.open(image_path).convert('RGBA')
            pixels = [p[:3] for p in img.getdata() if p[3] > 10]
            if not pixels:
                return (88, 166, 255)
            
            r = sum([p[0] for p in pixels]) // len(pixels)
            g = sum([p[1] for p in pixels]) // len(pixels)
            b = sum([p[2] for p in pixels]) // len(pixels)
            
            # Patlatılmış Renkler
            r = min(255, int(r * 1.5))
            g = min(255, int(g * 1.5))
            b = min(255, int(b * 1.5))
            return (r, g, b)
        except:
            return (88, 166, 255)

    def init_ui(self):
        self.setWindowTitle('Aircraft IFF Classifier')
        self.setFixedSize(920, 640)
        self.setStyleSheet("background: transparent;")

        wrapper_layout = QVBoxLayout(self)
        wrapper_layout.setContentsMargins(10, 10, 10, 10)

        # ŞEFFAF CAM ARKA PLAN
        self.main_container = QWidget()
        self.main_container.setStyleSheet('''
            background-color: rgba(13, 17, 23, 0.85);
            border: 1px solid #30363D;
            border-radius: 12px;
        ''')
        wrapper_layout.addWidget(self.main_container)

        main_layout = QVBoxLayout(self.main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- HEADER (SÜRÜKLENEBİLİR ALAN) ---
        header = QWidget()
        header.setFixedHeight(70)
        header.setStyleSheet('''
            background: transparent; 
            border-bottom: 1px solid rgba(48, 54, 61, 0.5);
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        ''')

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 20, 0)

        title = QLabel('Aircraft IFF Classifier ✈')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setStyleSheet('color: #FFFFFF; letter-spacing: 1px; border: none;')

        version = QLabel('v1.4 | Cyber-Tactical')
        version.setStyleSheet('color: #A3B3C1; font-size: 13px; font-weight: bold; border: none;')

        close_btn = QPushButton('✕')
        close_btn.setFixedSize(32, 32)
        close_btn.setStyleSheet('''
            QPushButton {
                background: transparent;
                color: #C9D1D9;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 16px;
            }
            QPushButton:hover { background: #F85149; color: white; }
            QPushButton:pressed { background: #B62324; }
        ''')
        close_btn.clicked.connect(self.close)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(version)
        header_layout.addSpacing(20)
        header_layout.addWidget(close_btn)

        main_layout.addWidget(header)

        # --- GÖVDE ---
        body = QWidget()
        body.setStyleSheet('background: transparent;')
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(30, 30, 30, 30)
        body_layout.setSpacing(25)

        # --- SOL PANEL ---
        self.left_panel = QWidget()
        self.left_panel.setObjectName("LeftPanel")
        self.left_panel.setFixedWidth(400)
        self.left_panel.setStyleSheet('''
            QWidget#LeftPanel {
                background: #161B22; 
                border: 1px solid #30363D;
                border-radius: 12px;
            }
        ''')

        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        left_title = QLabel('Radar Kaynağı (Görsel)')
        left_title.setFont(QFont('Arial', 14, QFont.Bold))
        left_title.setStyleSheet('color: #FFFFFF; border: none; background: transparent;')
        left_layout.addWidget(left_title)

        self.image_area = QLabel()
        self.image_area.setFixedHeight(280)
        self.image_area.setAlignment(Qt.AlignCenter)
        self.image_area.setText('Görsel seçmek için tıklayın')
        self.image_area.setStyleSheet('''
            QLabel {
                background: rgba(13, 17, 23, 200);
                border: 2px solid #30363D;
                border-radius: 10px;
                color: #A3B3C1;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel:hover {
                border-color: #58A6FF;
                color: #58A6FF;
            }
        ''')
        self.image_area.setCursor(Qt.PointingHandCursor)
        self.image_area.mousePressEvent = self.load_image
        left_layout.addWidget(self.image_area)

        self.load_btn = QPushButton('Görsel Seç')
        self.load_btn.setFixedHeight(44)
        self.load_btn.setStyleSheet('''
            QPushButton {
                background: #21262D;
                color: #E6EDF3;
                border: 1px solid #30363D;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: #30363D; border-color: #8B949E; }
            QPushButton:pressed { background: #161B22; }
        ''')
        self.load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_btn)
        left_layout.addStretch()

        # --- SAĞ PANEL ---
        self.right_panel = QWidget()
        self.right_panel.setObjectName("RightPanel") 
        self.right_panel.setStyleSheet('''
            QWidget#RightPanel {
                background: #161B22;
                border: 1px solid #30363D;
                border-radius: 12px;
            }
        ''')

        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        right_title = QLabel('Telemetri ve IFF Analizi')
        right_title.setFont(QFont('Arial', 14, QFont.Bold))
        right_title.setStyleSheet('color: #FFFFFF; border: none; background: transparent;')
        right_layout.addWidget(right_title)

        self.iff_card = QLabel('SİSTEM HAZIR')
        self.iff_card.setFixedHeight(85)
        self.iff_card.setAlignment(Qt.AlignCenter)
        self.iff_card.setFont(QFont('Arial', 17, QFont.Bold))
        self.iff_card.setStyleSheet('''
            background: #0D1117;
            border: 1px solid #30363D;
            border-radius: 10px;
            color: #A3B3C1;
            letter-spacing: 1px;
        ''')
        right_layout.addWidget(self.iff_card)

        self.confidence_label = QLabel('Güven skoru: —')
        self.confidence_label.setStyleSheet('color: #C9D1D9; font-size: 13px; font-weight: bold; border: none; background: transparent;')
        right_layout.addWidget(self.confidence_label)

        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet('background: #30363D; border: none;')
        right_layout.addWidget(line)

        self.bar_labels = {}
        self.bar_widgets = {}
        self.prob_labels = {}
        
        # Consolas fontu büyütüldü ve netleştirildi
        mono_font = QFont("Consolas", 12, QFont.Bold)
        mono_font.setStyleHint(QFont.Monospace)

        for name in LABEL_NAMES:
            row = QWidget()
            row.setStyleSheet('border: none; background: transparent;')
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(8)

            name_lbl = QLabel(name)
            name_lbl.setFixedWidth(65)
            name_lbl.setStyleSheet('color: #A3B3C1; font-size: 13px; font-weight: bold; border: none; background: transparent;')
            row_layout.addWidget(name_lbl)

            bar_bg = QWidget()
            bar_bg.setFixedHeight(14)
            bar_bg.setStyleSheet('background: #21262D; border-radius: 7px; border: none;')

            bar_fill = QWidget(bar_bg)
            bar_fill.setFixedHeight(14)
            bar_fill.setFixedWidth(0)
            bar_fill.setStyleSheet(f'background: {IFF_COLORS[name]}; border-radius: 7px;')

            row_layout.addWidget(bar_bg)

            prob_lbl = QLabel('% 0.0')
            prob_lbl.setFixedWidth(60)
            prob_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            prob_lbl.setFont(mono_font) 
            prob_lbl.setStyleSheet('color: #A3B3C1; border: none; background: transparent;')
            row_layout.addWidget(prob_lbl)

            right_layout.addWidget(row)

            self.bar_labels[name] = name_lbl
            self.bar_widgets[name] = (bar_bg, bar_fill)
            self.prob_labels[name] = prob_lbl

        right_layout.addStretch()

        # YENİ "TANIMLA" BUTONU (Cyber-Tactical)
        self.analyze_btn = QPushButton('TANIMLA')
        self.analyze_btn.setFixedHeight(50)
        self.analyze_btn.setStyleSheet('''
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1158C7, stop:1 #1D7AF2);
                color: #FFFFFF;
                border: 1px solid #388BFD;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 900;
                letter-spacing: 2px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1D7AF2, stop:1 #388BFD);
                border: 1px solid #58A6FF; 
            }
            QPushButton:pressed { 
                background: #0A5DC2; 
                border: 1px solid #1158C7;
            }
        ''')
        self.analyze_btn.clicked.connect(self.analyze)
        right_layout.addWidget(self.analyze_btn)

        body_layout.addWidget(self.left_panel)
        body_layout.addWidget(self.right_panel)
        main_layout.addWidget(body)

        self.show()

    # --- PENCERE SÜRÜKLEME ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_start_pos = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_active:
            self.move(event.globalPos() - self._drag_start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            event.accept()

    def load_image(self, _event=None):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Görsel Seç', '',
            'Görseller (*.png *.jpg *.jpeg)'
        )
        if path:
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(360, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_area.setPixmap(pixmap)
            self.image_area.setText('')
            self.current_image_path = path
            
            # --- SPOTIFY AMFİSİ (Hissedilir) ---
            r, g, b = self.get_dominant_color(path)
            
            self.left_panel.setStyleSheet(f'''
                QWidget#LeftPanel {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 rgba({r},{g},{b}, 0.5), 
                                stop:0.35 rgba({r},{g},{b}, 0.1), 
                                stop:1 #161B22);
                    border: 1px solid rgba({r},{g},{b}, 0.8);
                    border-radius: 12px;
                }}
            ''')
            
            self.right_panel.setStyleSheet('''
                QWidget#RightPanel {
                    background: #161B22;
                    border: 1px solid #30363D;
                    border-radius: 12px;
                }
            ''')
            
            self.iff_card.setText('BEKLENİYOR...')
            self.iff_card.setStyleSheet('''
                background: #0D1117;
                border: 1px solid #30363D;
                border-radius: 10px;
                color: #A3B3C1;
                letter-spacing: 2px;
            ''')
            self.confidence_label.setText('Güven skoru: —')
            for name in LABEL_NAMES:
                _, bar_fill = self.bar_widgets[name]
                bar_fill.setFixedWidth(0)
                self.prob_labels[name].setText('% 0.0')

    def analyze(self):
        if not self.current_image_path:
            return

        import tensorflow as tf

        if self.model is None:
            self.model = tf.keras.models.load_model('model/best_modelCLAHE.keras')

        img = Image.open(self.current_image_path).convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img_rgb = bg.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_rgb, dtype=np.uint8)

        img_array = np.array(ImageOps.autocontrast(Image.fromarray(img_array)), dtype=np.uint8)

        x = np.expand_dims(img_array.astype(np.float32), axis=0)
        proba = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(proba))
        confidence = proba[pred] * 100

        iff_text, iff_color, (r, g, b) = IFF_MAP[pred]
        aircraft_name = LABEL_NAMES[pred]
        
        if aircraft_name == 'Civil':
            aircraft_name = 'SİVİL HAVA ARACI'

        self.iff_card.setText(f'{aircraft_name}\n {iff_text}')
        self.iff_card.setStyleSheet(f'''
            background: #0D1117;
            border: 2px solid {iff_color};
            border-radius: 10px;
            color: {iff_color};
            font-weight: bold;
        ''')
        self.confidence_label.setText(f'Güven skoru: %{confidence:.1f}')

        # SAĞ PANEL İÇİN KATEGORİ AMFİSİ
        self.right_panel.setStyleSheet(f'''
            QWidget#RightPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                            stop:0 rgba({r},{g},{b}, 0.25), 
                            stop:1 #161B22);
                border: 1px solid {iff_color};
                border-radius: 12px;
            }}
        ''')

        for i, name in enumerate(LABEL_NAMES):
            prob = proba[i] * 100
            bar_bg, bar_fill = self.bar_widgets[name]
            bar_bg_width = bar_bg.width()
            fill_width = int((prob / 100) * (bar_bg_width if bar_bg_width > 0 else BAR_MAX_WIDTH))
            
            # --- GEOMETRİ HATASI ÇÖZÜMÜ ---
            # Genişlik 14px'ten küçükse şekil bozulur (kare/çizgi olur). 
            # Bu yüzden %0'dan büyük her değeri minimum 14px (tam bir yuvarlak) yapıyoruz.
            if fill_width > 0 and fill_width < 14:
                fill_width = 14
            elif prob == 0:
                fill_width = 0

            bar_fill.setFixedWidth(fill_width)
            self.prob_labels[name].setText(f'%{prob:5.1f}')

            # Okunabilirlik için seçili metinleri parlak BEYAZ yaptık
            if i == pred:
                self.bar_labels[name].setStyleSheet(
                    'color: #FFFFFF; font-size: 14px; font-weight: bold; border: none; background: transparent;'
                )
                self.prob_labels[name].setStyleSheet(
                    'color: #FFFFFF; font-weight: bold; border: none; background: transparent;'
                )
            else:
                self.bar_labels[name].setStyleSheet(
                    'color: #8B949E; font-size: 13px; font-weight: bold; border: none; background: transparent;'
                )
                self.prob_labels[name].setStyleSheet(
                    'color: #8B949E; border: none; background: transparent;'
                )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IFFApp()
    sys.exit(app.exec_())