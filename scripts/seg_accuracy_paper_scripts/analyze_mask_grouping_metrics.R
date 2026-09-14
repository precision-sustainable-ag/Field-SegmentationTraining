# ------------------------------------------------------------------------------
# Script: analyze_mask_grouping_metrics.R
#
# Purpose
#   Analyze per-image segmentation metrics and generate model-comparison box plots
#   across biological grouping schemes (e.g., group, family, morphology), plus
#   overall model-performance box plots.
#
# Input
#   CSV: data/.../mask_comparison_metrics/per_image_metrics_by_grouping.csv
#   Expected columns:
#     - model
#     - image_id
#     - grouping_type
#     - grouping
#     - metric columns such as:
#         fg_iou, fg_dice, precision, recall, mean_iou, mean_dice, accuracy
#
# What this script does
#   1) Validates required R packages.
#   2) Reads and validates the input CSV.
#   3) For each metric and grouping_type:
#      - builds black box plots of metric values by grouping, faceted by model
#      - runs ANOVA + Tukey's HSD within each model (p-value 0.05)
#      - converts Tukey adjusted p-values to compact letters (a, b, c, ...)
#      - remaps letters so highest-mean grouping gets "a"
#      - places letters at the bottom of each panel
#      - saves plots under: <out_dir>/<metric>/boxplot_<grouping_type>.png
#   4) For each metric, creates an overall model box plot (all images combined):
#      - runs ANOVA + Tukey's HSD across models
#      - assigns compact letters (highest-mean model gets "a")
#      - saves under: <out_dir>/overall_model_comparison/
#   5) Writes grouped mean summary table:
#      - <out_dir>/grouping_metric_means.csv
#
# Output directory
#   data/.../mask_comparison_metrics/r_model_performance_across_grouping
#
# Notes
#   - Y-axis limits are computed dynamically from observed values.
#   - Tukey letters indicate significance groups at alpha_sig (default 0.05):
#       groups sharing a letter are not significantly different.
# ------------------------------------------------------------------------------

# Load required packages (no auto-install)
required_packages <- c("ggplot2", "dplyr", "viridis", "readr", "multcompView")

missing_pkgs <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing required packages: ",
    paste(missing_pkgs, collapse = ", "),
    ". Please install them before running this script."
  )
}

for (pkg in required_packages) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE, warn.conflicts = FALSE)
  )
}

base_dir <- "data/nav_seg_model_accuracy_comp_paper/data/test_dataset/images_masks_grouped/all_images_masks_in_grouping"
metrics_dir <- file.path(base_dir, "mask_comparison_metrics")
grouped_metrics_csv <- file.path(metrics_dir, "per_image_metrics_by_grouping.csv")
out_dir <- file.path(metrics_dir, "r_model_performance_across_grouping")
alpha_sig <- 0.05

safe_name <- function(x) {
  x <- as.character(x)
  x <- gsub("[^[:alnum:]_\\-]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  if (nchar(x) == 0) "unnamed" else x
}

compute_dynamic_ylim <- function(values, hard_min = 0, hard_max = 1, pad_frac = 0.12, min_span = 0.05, min_pad = 0.01) {
  vals <- values[is.finite(values)]
  if (length(vals) == 0) return(c(hard_min, hard_max))

  vmin <- min(vals)
  vmax <- max(vals)
  span <- vmax - vmin
  pad <- if (span <= 0) max(min_pad, abs(vmax) * 0.01) else max(span * pad_frac, min_pad)

  y_min <- max(hard_min, vmin - pad)
  y_max <- min(hard_max, vmax + pad)

  if ((y_max - y_min) < min_span) {
    mid <- (y_min + y_max) / 2
    y_min <- max(hard_min, mid - min_span / 2)
    y_max <- min(hard_max, mid + min_span / 2)
  }

  if (y_min >= y_max) {
    y_min <- max(hard_min, vmin - min_pad)
    y_max <- min(hard_max, vmax + min_pad)
  }

  c(y_min, y_max)
}

colorful_theme <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "#F7FBFF", color = NA),
      panel.background = ggplot2::element_rect(fill = "#EEF4FF", color = NA),
      panel.grid.major = ggplot2::element_line(color = "#D3E1F2", linewidth = 0.35),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1), # parallel to y-axis
      strip.text = ggplot2::element_text(face = "bold", color = "#1F3B73"),
      plot.title = ggplot2::element_text(face = "bold", color = "#1F3B73", hjust = 0.5) # centered
    )
}

# Remap CLD symbols so highest mean contains "a", then "b", etc.
remap_letters_by_mean <- function(letters_named, means_df) {
  if (length(letters_named) == 0) return(letters_named)

  letter_pool <- c(letters, LETTERS)
  seen_old <- character(0)
  ordered_groups <- as.character(means_df$grouping)

  for (g in ordered_groups) {
    if (!g %in% names(letters_named)) next
    lbl <- letters_named[[g]]
    if (is.null(lbl) || is.na(lbl) || lbl == "") next

    chars <- unique(strsplit(lbl, "", fixed = TRUE)[[1]])
    seen_old <- c(seen_old, chars[!chars %in% seen_old])
  }

  seen_old <- unique(seen_old)
  if (length(seen_old) == 0) return(letters_named)

  if (length(seen_old) > length(letter_pool)) {
    stop("Too many significance groups to map into letters.")
  }

  map <- stats::setNames(letter_pool[seq_along(seen_old)], seen_old)

  remapped <- vapply(names(letters_named), function(g) {
    lbl <- letters_named[[g]]
    if (is.null(lbl) || is.na(lbl) || lbl == "") return("")

    chars <- unique(strsplit(lbl, "", fixed = TRUE)[[1]])
    mapped <- unname(map[chars])
    mapped <- mapped[!is.na(mapped)]
    mapped <- mapped[order(match(mapped, letter_pool))]
    paste0(mapped, collapse = "")
  }, FUN.VALUE = character(1))

  stats::setNames(remapped, names(letters_named))
}

# Build compact-letter display per model using Tukey's HSD
build_letters_df <- function(sub_df, metric_name, alpha = 0.05) {
  model_levels <- levels(droplevels(sub_df$model))
  if (is.null(model_levels) || length(model_levels) == 0) {
    model_levels <- sort(unique(as.character(sub_df$model)))
  }

  cld_rows <- list()

  for (m in model_levels) {
    md <- sub_df |>
      dplyr::filter(as.character(model) == m) |>
      dplyr::transmute(
        grouping = droplevels(as.factor(grouping)),
        value = as.numeric(.data[[metric_name]])
      ) |>
      dplyr::filter(is.finite(value), !is.na(grouping))

    if (nrow(md) < 2 || nlevels(md$grouping) < 2) {
      next
    }

    fit <- try(stats::aov(value ~ grouping, data = md), silent = TRUE)
    if (inherits(fit, "try-error")) {
      next
    }

    # Explicit Tukey HSD
    tk <- try(stats::TukeyHSD(fit, "grouping"), silent = TRUE)
    if (inherits(tk, "try-error") || !("grouping" %in% names(tk))) {
      next
    }

    tk_df <- as.data.frame(tk$grouping)
    if (nrow(tk_df) == 0) {
      next
    }

    p_col <- if ("p adj" %in% colnames(tk_df)) {
      "p adj"
    } else if ("p.adj" %in% colnames(tk_df)) {
      "p.adj"
    } else {
      NA_character_
    }

    if (is.na(p_col)) {
      next
    }

    p_vals <- as.numeric(tk_df[[p_col]])
    names(p_vals) <- rownames(tk_df)

    cld <- try(
      multcompView::multcompLetters(
        p_vals,
        threshold = alpha,
        Letters = c(letters, LETTERS)
      ),
      silent = TRUE
    )

    if (inherits(cld, "try-error") || is.null(cld$Letters)) {
      next
    }

    # Means for ordering letters (highest mean -> "a")
    means_df <- md |>
      dplyr::group_by(grouping) |>
      dplyr::summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") |>
      dplyr::arrange(dplyr::desc(mean_value))

    letters_named <- remap_letters_by_mean(cld$Letters, means_df)

    cld_df <- data.frame(
      grouping = names(letters_named),
      letter = as.character(letters_named),
      model = factor(m, levels = model_levels),
      stringsAsFactors = FALSE
    )

    cld_rows[[length(cld_rows) + 1]] <- cld_df
  }

  if (length(cld_rows) == 0) {
    return(data.frame(
      grouping = character(),
      letter = character(),
      model = factor(character(), levels = levels(sub_df$model)),
      stringsAsFactors = FALSE
    ))
  }

  dplyr::bind_rows(cld_rows)
}

# Build letters for one factor (e.g., model) using ANOVA + Tukey HSD
build_letters_one_factor <- function(input_df, value_col, factor_col, alpha = 0.05) {
  empty <- data.frame(grouping = character(), letter = character(), stringsAsFactors = FALSE)

  if (!(value_col %in% colnames(input_df)) || !(factor_col %in% colnames(input_df))) {
    return(empty)
  }

  td <- input_df |>
    dplyr::transmute(
      grouping = droplevels(as.factor(.data[[factor_col]])),
      value = as.numeric(.data[[value_col]])
    ) |>
    dplyr::filter(is.finite(value), !is.na(grouping))

  if (nrow(td) < 2 || nlevels(td$grouping) < 2) {
    return(empty)
  }

  fit <- try(stats::aov(value ~ grouping, data = td), silent = TRUE)
  if (inherits(fit, "try-error")) return(empty)

  tk <- try(stats::TukeyHSD(fit, "grouping"), silent = TRUE)
  if (inherits(tk, "try-error") || !("grouping" %in% names(tk))) return(empty)

  tk_df <- as.data.frame(tk$grouping)
  if (nrow(tk_df) == 0) return(empty)

  p_col <- if ("p adj" %in% colnames(tk_df)) {
    "p adj"
  } else if ("p.adj" %in% colnames(tk_df)) {
    "p.adj"
  } else {
    NA_character_
  }
  if (is.na(p_col)) return(empty)

  p_vals <- as.numeric(tk_df[[p_col]])
  names(p_vals) <- rownames(tk_df)

  cld <- try(
    multcompView::multcompLetters(
      p_vals,
      threshold = alpha,
      Letters = c(letters, LETTERS)
    ),
    silent = TRUE
  )
  if (inherits(cld, "try-error") || is.null(cld$Letters)) return(empty)

  means_df <- td |>
    dplyr::group_by(grouping) |>
    dplyr::summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(mean_value))

  # highest mean -> "a", next -> "b", etc.
  letters_named <- remap_letters_by_mean(cld$Letters, means_df)

  data.frame(
    grouping = names(letters_named),
    letter = as.character(letters_named),
    stringsAsFactors = FALSE
  )
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(grouped_metrics_csv)) {
  stop("Missing input CSV: ", grouped_metrics_csv)
}

df <- readr::read_csv(grouped_metrics_csv, show_col_types = FALSE)

# Expected columns
required_cols <- c("model", "image_id", "grouping_type", "grouping")
metric_candidates <- c("fg_iou", "fg_dice", "precision", "recall", "mean_iou", "mean_dice", "accuracy")

missing_required <- setdiff(required_cols, colnames(df))
if (length(missing_required) > 0) {
  stop("Missing required columns: ", paste(missing_required, collapse = ", "))
}

metrics_to_plot <- intersect(metric_candidates, colnames(df))
if (length(metrics_to_plot) == 0) {
  stop("No metric columns found. Expected one of: ", paste(metric_candidates, collapse = ", "))
}

df <- df |>
  dplyr::mutate(
    model = as.factor(model),
    grouping_type = as.factor(grouping_type),
    grouping = as.factor(grouping)
  )

# Save summary table
summary_df <- df |>
  dplyr::group_by(model, grouping_type, grouping) |>
  dplyr::summarise(
    dplyr::across(dplyr::all_of(metrics_to_plot), ~ mean(.x, na.rm = TRUE)),
    n = dplyr::n(),
    .groups = "drop"
  )

readr::write_csv(summary_df, file.path(out_dir, "grouping_metric_means.csv"))

for (metric_name in metrics_to_plot) {
  metric_dir <- file.path(out_dir, safe_name(metric_name))
  dir.create(metric_dir, recursive = TRUE, showWarnings = FALSE)

  for (gt_name in sort(unique(as.character(df$grouping_type)))) {
    gt_slug <- safe_name(gt_name)
    axis_label <- gsub("_", " ", gt_name)
    plot_title <- paste(metric_name, "by", axis_label)

    sub_df <- df |>
      dplyr::filter(as.character(grouping_type) == gt_name) |>
      dplyr::filter(!is.na(.data[[metric_name]])) |>
      dplyr::mutate(
        grouping = droplevels(as.factor(grouping)),
        model = droplevels(as.factor(model))
      )

    if (nrow(sub_df) == 0 || nlevels(sub_df$grouping) == 0) {
      next
    }

    y_limits <- compute_dynamic_ylim(sub_df[[metric_name]])
    y_span <- y_limits[2] - y_limits[1]

    letters_df <- build_letters_df(sub_df, metric_name, alpha = alpha_sig) |>
      dplyr::mutate(
        grouping = factor(grouping, levels = levels(sub_df$grouping)),
        y_pos = y_limits[1] + (0.03 * y_span)  # bottom of panel
      )

    p_box <- ggplot2::ggplot(
      sub_df,
      ggplot2::aes(x = grouping, y = .data[[metric_name]])
    ) +
      ggplot2::geom_boxplot(
        fill = "black",
        color = "black",
        alpha = 0.7,
        outlier.alpha = 0.7,
        outlier.color = "black",
        width = 0.65,
        linewidth = 0.35
      ) +
      ggplot2::facet_wrap(~ model, scales = "fixed") +
      ggplot2::coord_cartesian(ylim = y_limits, expand = FALSE, clip = "off") +
      ggplot2::labs(
        title = plot_title,  # e.g., "fg_dice by morphology"
        x = axis_label,      # x-axis label is grouping type name
        y = metric_name
      ) +
      colorful_theme()

    if (nrow(letters_df) > 0) {
      p_box <- p_box +
        ggplot2::geom_text(
          data = letters_df,
          ggplot2::aes(x = grouping, y = y_pos, label = letter),
          inherit.aes = FALSE,
          vjust = 0,
          size = 3.8,
          fontface = "bold",
          color = "#1F3B73"
        )
    }

    ggplot2::ggsave(
      filename = file.path(metric_dir, paste0("boxplot_", gt_slug, ".png")),
      plot = p_box,
      width = 12,
      height = 7,
      dpi = 150
    )
  }
}

# Additional overall model comparison plots (across all images)
overall_dir <- file.path(out_dir, "overall_model_comparison")
dir.create(overall_dir, recursive = TRUE, showWarnings = FALSE)

for (metric_name in metrics_to_plot) {
  overall_df <- df |>
    dplyr::filter(!is.na(.data[[metric_name]])) |>
    dplyr::mutate(model = droplevels(as.factor(model)))

  if (nrow(overall_df) == 0 || nlevels(overall_df$model) < 2) {
    next
  }

  # order models by overall mean (high -> low)
  model_order <- overall_df |>
    dplyr::group_by(model) |>
    dplyr::summarise(mean_value = mean(.data[[metric_name]], na.rm = TRUE), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(mean_value))

  overall_df <- overall_df |>
    dplyr::mutate(model = factor(as.character(model), levels = as.character(model_order$model)))

  y_limits <- compute_dynamic_ylim(overall_df[[metric_name]])
  y_span <- y_limits[2] - y_limits[1]

  letters_df_model <- build_letters_one_factor(
    input_df = overall_df,
    value_col = metric_name,
    factor_col = "model",
    alpha = alpha_sig
  ) |>
    dplyr::mutate(
      model = factor(grouping, levels = levels(overall_df$model)),
      y_pos = y_limits[1] + (0.03 * y_span)
    ) |>
    dplyr::filter(!is.na(model))

  p_overall <- ggplot2::ggplot(
    overall_df,
    ggplot2::aes(x = model, y = .data[[metric_name]])
  ) +
    ggplot2::geom_boxplot(
      fill = "black",
      color = "black",
      alpha = 0.7,
      outlier.alpha = 0.7,
      outlier.color = "black",
      width = 0.65,
      linewidth = 0.35
    ) +
    ggplot2::coord_cartesian(ylim = y_limits, expand = FALSE, clip = "off") +
    ggplot2::labs(
      title = paste(metric_name, "by model"),
      x = "model",
      y = metric_name
    ) +
    colorful_theme()

  if (nrow(letters_df_model) > 0) {
    p_overall <- p_overall +
      ggplot2::geom_text(
        data = letters_df_model,
        ggplot2::aes(x = model, y = y_pos, label = letter),
        inherit.aes = FALSE,
        vjust = 0,
        size = 3.8,
        fontface = "bold",
        color = "#1F3B73"
      )
  }

  ggplot2::ggsave(
    filename = file.path(overall_dir, paste0("boxplot_model_", safe_name(metric_name), ".png")),
    plot = p_overall,
    width = 12,
    height = 7,
    dpi = 150
  )
}

cat("Done. Grouping-wise and overall model box plots with Tukey-HSD letters written to:\n", out_dir, "\n")