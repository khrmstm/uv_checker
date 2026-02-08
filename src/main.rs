use egui::ColorImage;
use image::RgbaImage;

use image::{GrayImage, ImageReader, Luma, Rgba};
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate_mut;
use imageproc::region_labelling::{connected_components, Connectivity};
use std::collections::HashMap;
use std::path::Path;

use eframe::{egui, egui::TextureHandle};

//MX-linux need gtk3 issue for rfd.
//run below command.
//sudo apt install xdg-desktop-portal xdg-desktop-portal-gtk

use rfd;
use std::path::PathBuf;

//RgbaImage → egui::ColorImage 変換ヘルパ
fn rgba_image_to_color_image(img: &RgbaImage) -> ColorImage {
  let (w, h) = img.dimensions();
  let size = [w as usize, h as usize];
  let flat = img.as_flat_samples();
  ColorImage::from_rgba_unmultiplied(size, flat.as_slice())
}

//UV チェック処理

fn analyze_uv<P: AsRef<Path>>(path: P) -> Result<RgbaImage, Box<dyn std::error::Error>> {
  let img = ImageReader::open(path)?.decode()?.to_rgba8();
  let (width, height) = img.dimensions();
  let mut img_grey = GrayImage::new(width, height);

  // 透明ピクセル以外を前景(255)に変換
  for (x, y, pixel) in img.enumerate_pixels() {
    let alpha = pixel[3];
    img_grey.put_pixel(x, y, Luma([if alpha > 128 { 255u8 } else { 0u8 }]));
  }

  // 連結成分ラベリング（8近傍）
  let background = Luma([0u8]);
  let labeled = connected_components(&img_grey, Connectivity::Eight, background);

  // 各ラベル領域の境界ボックス + ピクセルカウント
  let mut label_info: HashMap<u32, ((u32, u32), (u32, u32), u32)> = HashMap::new();

  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    if label == 0 {
      continue;
    }
    let entry = label_info
      .entry(label)
      .or_insert(((u32::MAX, u32::MAX), (0, 0), 0));
    entry.0 .0 = entry.0 .0.min(x);
    entry.0 .1 = entry.0 .1.min(y);
    entry.1 .0 = entry.1 .0.max(x);
    entry.1 .1 = entry.1 .1.max(y);
    entry.2 += 1;
  }

  let mut result_img = RgbaImage::new(width, height);

  let mut label_status: HashMap<u32, (bool, bool)> = HashMap::new();
  let labels: Vec<u32> = label_info.keys().copied().collect();

  let margin_threshold = 8u8;

  // ラベルごとのマスク
  let mut label_masks: HashMap<u32, GrayImage> = HashMap::new();
  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    if label == 0 {
      continue;
    }
    let mask = label_masks
      .entry(label)
      .or_insert_with(|| GrayImage::new(width, height));
    mask.put_pixel(x, y, Luma([255u8]));
  }

  // 膨張
  let mut dilated_masks: HashMap<u32, GrayImage> = HashMap::new();
  for (label, mask) in &label_masks {
    let mut dilated = mask.clone();
    dilate_mut(&mut dilated, Norm::LInf, margin_threshold);
    dilated_masks.insert(*label, dilated);
  }

  // ラベルごとの margin チェック
  for &label in &labels {
    let ((min_x, min_y), (max_x, max_y), pixel_count) = label_info[&label];
    let bbox_area =
      (max_x.saturating_sub(min_x) + 1) as u64 * (max_y.saturating_sub(min_y) + 1) as u64;

    let is_problem = pixel_count * 3 < bbox_area as u32;

    let dilated_self = &dilated_masks[&label];
    let mut margin_ok = true;

    for &other_label in &labels {
      if other_label == label {
        continue;
      }
      let dilated_other = &dilated_masks[&other_label];

      let mut overlap = false;
      for y in 0..height {
        for x in 0..width {
          let v1 = dilated_self.get_pixel(x, y)[0];
          let v2 = dilated_other.get_pixel(x, y)[0];
          if v1 > 0 && v2 > 0 {
            overlap = true;
            break;
          }
        }
        if overlap {
          break;
        }
      }
      if overlap {
        margin_ok = false;
        break;
      }
    }

    label_status.insert(label, (is_problem, margin_ok));
  }

  // margin マスク
  let mut margin_mask = GrayImage::new(width, height);
  for dilated in dilated_masks.values() {
    for y in 0..height {
      for x in 0..width {
        let v = dilated.get_pixel(x, y)[0];
        if v > 0 {
          margin_mask.put_pixel(x, y, Luma([255u8]));
        }
      }
    }
  }

  // conflict マスク
  let mut conflict_mask = GrayImage::new(width, height);
  for &label in &labels {
    let dilated_self = &dilated_masks[&label];
    for &other_label in &labels {
      if other_label == label {
        continue;
      }
      let dilated_other = &dilated_masks[&other_label];
      for y in 0..height {
        for x in 0..width {
          let v1 = dilated_self.get_pixel(x, y)[0];
          let v2 = dilated_other.get_pixel(x, y)[0];
          if v1 > 0 && v2 > 0 {
            conflict_mask.put_pixel(x, y, Luma([255u8]));
          }
        }
      }
    }
  }

  // 結果画像塗り分け
  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    let margin = margin_mask.get_pixel(x, y)[0] > 0;
    let conflict = conflict_mask.get_pixel(x, y)[0] > 0;

    let on_border = x == 0 || x == width - 1 || y == 0 || y == height - 1;
    let on_border_margin: bool = margin && on_border;

    let out_px = if on_border_margin {
      Rgba([128, 0, 128, 255]) // violet
    } else if label == 0 {
      if conflict {
        Rgba([255, 0, 0, 255]) // red
      } else if margin {
        Rgba([0, 255, 255, 255]) // cyan
      } else {
        Rgba([0, 0, 0, 0]) // transparent
      }
    } else {
      if conflict {
        Rgba([255, 0, 0, 255]) // red
      } else if margin {
        Rgba([0, 255, 255, 255]) // cyan
      } else {
        Rgba([0, 255, 0, 255]) // green
      }
    };

    result_img.put_pixel(x, y, out_px);
  }

  Ok(result_img)
}

//eframe/egui アプリ本体

struct UvApp {
  uv_path: Option<PathBuf>,
  result_texture: Option<TextureHandle>,
  status: String,
  zoom: f32,
}

impl Default for UvApp {
  fn default() -> Self {
    Self {
      uv_path: None,
      result_texture: None,
      status: "No file loaded".to_string(),
      zoom: 1.0, //100%
    }
  }
}

impl eframe::App for UvApp {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.horizontal(|ui| {
        if ui.button("Load image").clicked() {
          if let Some(path) = rfd::FileDialog::new().pick_file() {
            self.uv_path = Some(path.clone());
            self.status = "Analyzing...".to_string();

            match analyze_uv(&path) {
              Ok(result_img) => {
                let color_img = rgba_image_to_color_image(&result_img);
                let tex = ctx.load_texture("uv_result", color_img, Default::default());
                self.result_texture = Some(tex);
                self.status = "Done".to_string();
              }
              Err(e) => {
                self.result_texture = None;
                self.status = format!("Error: {e}");
              }
            }
          }
        }

        if let Some(path) = &self.uv_path {
          ui.label(path.to_string_lossy());
        } else {
          ui.label("No image selected");
        }
      });

      ui.label(&self.status);
      ui.separator();

      ui.horizontal(|ui| {
        if ui.button("−").clicked() {
          self.zoom *= 0.9;
        }
        if ui.button("+").clicked() {
          self.zoom *= 1.1;
        }
        ui.label(format!("{:.0}x", self.zoom));
      });

      if let Some(tex) = &self.result_texture {
        egui::ScrollArea::both()
          .max_height(ui.available_height())
          .show(ui, |ui| {
            let desired_size = tex.size_vec2() * self.zoom;
            let response = ui.image(&*tex);

            if response.hovered() {
              let scroll = ui.input(|i| i.raw_scroll_delta.y);
              if scroll.abs() > 0.0 {
                let factor = if scroll > 0.0 { 1.1 } else { 0.9 };
                self.zoom = (self.zoom * factor).clamp(0.1, 20.0);
              }
            }
          });
      } else {
        ui.label("Result image not available");
      }
    });
  }
}

fn main() -> Result<(), eframe::Error> {
  let options = eframe::NativeOptions::default();
  eframe::run_native(
    "UV Margin Checker",
    options,
    Box::new(|_cc| Ok(Box::new(UvApp::default()))), // Add Ok()
  )
}
