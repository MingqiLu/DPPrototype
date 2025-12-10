import os
import pandas as pd
from plotnine import (
    ggplot, aes,
    geom_point, geom_line,
    facet_wrap, facet_grid,
    theme_classic, theme,
    labs, scale_x_discrete
)

#==============================================================
# 0. 读取数据 & 重命名列去空格
#==============================================================

df = pd.read_csv("User Study Result.csv", skiprows=1)

df = df.rename(columns={
    "Tool Used": "Tool",
    "Baseline Knowledge": "Baseline"
})

# 把 Participant ID 转成字符串/类别，方便离散上色
df["Participant ID"] = df["Participant ID"].astype(str)

#--------------------------------------------------------------
# 转成长表 Pre/Post total scores
#--------------------------------------------------------------
total_long = df.melt(
    id_vars=["Participant ID", "Tool", "Baseline"],
    value_vars=["Pretest Score", "Posttest Score"],
    var_name="when_raw", value_name="score"
)
total_long["when"] = total_long["when_raw"].map({
    "Pretest Score": "Pre",
    "Posttest Score": "Post"
})
total_long = total_long.drop(columns=["when_raw"])
total_long["when"] = pd.Categorical(total_long["when"], ["Pre", "Post"], ordered=True)

#==============================================================
# 1. 每个人 Pre→Post 总分变化
#==============================================================

plot1 = (
    ggplot(total_long,
           aes("when", "score",
               group="Participant ID",
               color="Participant ID"))
    + geom_line(alpha=0.5)
    + geom_point(size=2)
    + labs(title="(1) Subject-wise Total Score Change",
           x="", y="Total Score", color="Participant ID")
    + scale_x_discrete(limits=["Pre", "Post"])
    + theme_classic()
    + theme(figure_size=(8, 6))
)

#==============================================================
# 2. 按 Tool 分类的每人总分变化
#==============================================================

plot2 = (
    ggplot(total_long,
           aes("when", "score",
               group="Participant ID",
               color="Participant ID"))
    + geom_line(alpha=0.5)
    + geom_point(size=2)
    + facet_wrap("~ Tool")
    + labs(title="(2) Score Change by Tool",
           x="", y="Total Score", color="Participant ID")
    + scale_x_discrete(limits=["Pre", "Post"])
    + theme_classic()
    + theme(figure_size=(10, 6))
)

#==============================================================
# 3. 题组分析 Group1/Group2
#   Group 1 = Pre[1,2,4] ↔ Post[3,6,7]
#   Group 2 = 其它题目
#==============================================================

pre_group1 = ["Pre1", "Pre2", "Pre4"]
post_group1 = ["Post3", "Post6", "Post7"]

all_pre  = [f"Pre{i}" for i in range(1, 8)]
all_post = [f"Post{i}" for i in range(1, 8)]

pre_group2  = sorted(list(set(all_pre) - set(pre_group1)))
post_group2 = sorted(list(set(all_post) - set(post_group1)))

df_groups = df.copy()
df_groups["Pre_group1"]  = df_groups[pre_group1].sum(axis=1)
df_groups["Post_group1"] = df_groups[post_group1].sum(axis=1)
df_groups["Pre_group2"]  = df_groups[pre_group2].sum(axis=1)
df_groups["Post_group2"] = df_groups[post_group2].sum(axis=1)

group_long = df_groups.melt(
    id_vars=["Participant ID", "Tool", "Baseline"],
    value_vars=["Pre_group1", "Post_group1", "Pre_group2", "Post_group2"],
    var_name="tmp", value_name="score"
)

group_long[["when", "group"]] = group_long["tmp"].str.extract(r"(Pre|Post)_group([12])")
group_long = group_long.drop(columns=["tmp"])
group_long["group"] = group_long["group"].map({"1": "Group 1", "2": "Group 2"})
group_long["when"]  = pd.Categorical(group_long["when"], ["Pre", "Post"], ordered=True)

plot3 = (
    ggplot(group_long,
           aes("when", "score",
               group="Participant ID",
               color="Participant ID"))
    + geom_line(alpha=0.5)
    + geom_point(size=2)
    + facet_grid("group ~ Tool")
    + labs(title="(3) Group1/2 Question Score Change by Tool",
           x="", y="Group Score", color="Participant ID")
    + scale_x_discrete(limits=["Pre", "Post"])
    + theme_classic()
    + theme(figure_size=(12, 6))
)

#==============================================================
# 4. Baseline × Tool × 每人变化
#==============================================================

plot4 = (
    ggplot(total_long,
           aes("when", "score",
               group="Participant ID",
               color="Participant ID"))
    + geom_line(alpha=0.5)
    + geom_point(size=2)
    + facet_grid("Baseline ~ Tool")
    + labs(title="(4) Score Change by Baseline × Tool",
           x="", y="Total Score", color="Participant ID")
    + scale_x_discrete(limits=["Pre", "Post"])
    + theme_classic()
    + theme(figure_size=(12, 8))
)

#==============================================================
# 保存所有图片
#==============================================================

os.makedirs("plots", exist_ok=True)

plot1.save("plots/1_total_change.png", dpi=300)
plot2.save("plots/2_tool_total_change.png", dpi=300)
plot3.save("plots/3_group_change_by_tool.png", dpi=300)
plot4.save("plots/4_baseline_tool_change.png", dpi=300)

print("\n🎉 所有图已保存到 ./plots/ 目录中")
